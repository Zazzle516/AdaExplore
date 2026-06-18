import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wd, stride_wh, stride_ww,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    M, N_, K,
    OHW, OW_,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # spatial+batch tile (M dim)
    pid_n = tl.program_id(1)  # OC tile (N dim)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                    # [BLOCK_K]

    # decode m -> (n_idx, od, oh, ow)
    n_idx = offs_m // (OD * OHW)
    rem = offs_m % (OD * OHW)
    od = rem // OHW
    rem2 = rem % OHW
    oh = rem2 // OW_
    ow = rem2 % OW_

    m_mask = offs_m < M
    n_mask = offs_n < N_

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # base input offset per row (no padding since padding=0)
    x_base = n_idx * stride_xn + od * stride_xd + oh * stride_xh + ow * stride_xw  # [BLOCK_M]

    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + offs_k  # [BLOCK_K]
        k_mask = k_idx < K
        # decode k -> (ic, kd, kh, kw)
        ic = k_idx // (KD * KH * KW)
        krem = k_idx % (KD * KH * KW)
        kd = krem // (KH * KW)
        krem2 = krem % (KH * KW)
        kh = krem2 // KW
        kw = krem2 % KW

        # x offsets: [BLOCK_M, BLOCK_K]
        x_off = (x_base[:, None]
                 + ic[None, :] * stride_xc
                 + kd[None, :] * stride_xd
                 + kh[None, :] * stride_xh
                 + kw[None, :] * stride_xw)
        x_mask = m_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

        # w offsets: [BLOCK_K, BLOCK_N], W is [OC, IC, KD, KH, KW]
        w_off = (offs_n[None, :] * stride_wo
                 + ic[:, None] * stride_wi
                 + kd[:, None] * stride_wd
                 + kh[:, None] * stride_wh
                 + kw[:, None] * stride_ww)
        w_mask = k_mask[:, None] & n_mask[None, :]
        w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    # bias
    b = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + b[None, :]

    # Mish: x * tanh(softplus(x))
    sp = tl.log(1.0 + tl.exp(acc))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    mish = acc * th
    # tanh
    e2m = tl.exp(2.0 * mish)
    out_val = (e2m - 1.0) / (e2m + 1.0)

    # store: out[n_idx, oc, od, oh, ow]
    out_off = (n_idx[:, None] * stride_on
               + offs_n[None, :] * stride_oc
               + od[:, None] * stride_od
               + oh[:, None] * stride_oh
               + ow[:, None] * stride_ow)
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_off, out_val, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kd = self.kh = self.kw = kernel_size
        else:
            self.kd, self.kh, self.kw = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)

    def forward(self, x):
        # only fast-path: stride=1, padding=0
        if (self.stride == (1, 1, 1) and self.padding == (0, 0, 0)
                and x.is_cuda and x.dtype == torch.float32):
            x = x.contiguous()
            w = self.conv.weight.contiguous()
            b = self.conv.bias.contiguous()
            N, IC, ID, IH, IW = x.shape
            OC = self.out_channels
            KD, KH, KW = self.kd, self.kh, self.kw
            OD = ID - KD + 1
            OH = IH - KH + 1
            OW = IW - KW + 1
            out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

            M = N * OD * OH * OW
            N_ = OC
            K = IC * KD * KH * KW
            OHW = OH * OW

            grid = lambda meta: (
                triton.cdiv(M, meta['BLOCK_M']),
                triton.cdiv(N_, meta['BLOCK_N']),
            )

            conv3d_fused_kernel[grid](
                x, w, b, out,
                N, IC, ID, IH, IW,
                OC, OD, OH, OW,
                KD, KH, KW,
                x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
                w.stride(0), w.stride(1), w.stride(2), w.stride(3), w.stride(4),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
                M, N_, K,
                OHW, OW,
            )
            return out

        # fallback
        x = self.conv(x)
        x = F.mish(x)
        x = torch.tanh(x)
        return x