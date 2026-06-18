import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv3d_implicit_gemm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    M, K_total, N_dim,
    DHW, HW,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = N * OD * OH * OW (rows)
    # N_dim = OC (cols)
    # K_total = IC * KD * KH * KW
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Decompose row index -> (n, od, oh, ow)
    # m -> n * OD*OH*OW + od*OH*OW + oh*OW + ow
    n_idx = offs_m // DHW
    rem = offs_m % DHW
    od_idx = rem // HW
    rem2 = rem % HW
    oh_idx = rem2 // OW
    ow_idx = rem2 % OW

    m_mask = offs_m < M
    n_mask = offs_n < N_dim

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    KHW = KH * KW
    KDHW = KD * KHW

    # iterate over K dimension
    for k_start in range(0, K_total, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]
        k_mask = k_idx < K_total

        # decompose k -> (ic, kd, kh, kw)
        ic = k_idx // KDHW
        krem = k_idx % KDHW
        kd = krem // KHW
        krem2 = krem % KHW
        kh = krem2 // KW
        kw = krem2 % KW

        # Input offsets:
        # x[n, ic, od+kd, oh+kh, ow+kw]  (padding=0, stride=1)
        # shape [BLOCK_M, BLOCK_K]
        in_d = od_idx[:, None] + kd[None, :]
        in_h = oh_idx[:, None] + kh[None, :]
        in_w = ow_idx[:, None] + kw[None, :]

        x_offset = (n_idx[:, None] * (IC * ID * IH * IW)
                    + ic[None, :] * (ID * IH * IW)
                    + in_d * (IH * IW)
                    + in_h * IW
                    + in_w)

        x_mask = m_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)

        # Weight offsets: w[oc, ic, kd, kh, kw], laid out (OC, K_total)
        w_offset = offs_n[:, None] * K_total + k_idx[None, :]
        w_mask = n_mask[:, None] & k_mask[None, :]
        w_vals = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)  # [BLOCK_N, BLOCK_K]

        acc += tl.dot(x_vals, tl.trans(w_vals))

    # bias
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc += bias[None, :]

    # Mish: x * tanh(softplus(x))
    sp = tl.log(1.0 + tl.exp(acc))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    m_val = acc * th
    # tanh
    e2b = tl.exp(2.0 * m_val)
    out = (e2b - 1.0) / (e2b + 1.0)

    # Store: out[n, oc, od, oh, ow]
    # output is NCDHW: index = n*OC*OD*OH*OW + oc*OD*OH*OW + od*OH*OW + oh*OW + ow
    out_offset = (n_idx[:, None] * (OC * DHW)
                  + offs_n[None, :] * DHW
                  + od_idx[:, None] * HW
                  + oh_idx[:, None] * OW
                  + ow_idx[:, None])
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_offset, out, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kD = self.kH = self.kW = kernel_size
        else:
            self.kD, self.kH, self.kW = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Fall back to torch if stride/padding not (1,0)
        if self.stride != 1 or self.padding != 0:
            x = self.conv(x)
            x = F.mish(x)
            x = torch.tanh(x)
            return x

        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        kD, kH, kW = self.kD, self.kH, self.kW
        OD = ID - kD + 1
        OH = IH - kH + 1
        OW = IW - kW + 1

        weight = self.conv.weight.contiguous().view(OC, IC * kD * kH * kW)
        bias = self.conv.bias.contiguous()

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        M = N * OD * OH * OW
        K_total = IC * kD * kH * kW
        N_dim = OC
        DHW = OD * OH * OW
        HW = OH * OW

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N_dim, meta['BLOCK_N']),
        )

        conv3d_implicit_gemm_kernel[grid](
            x, weight, bias, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            kD, kH, kW,
            M, K_total, N_dim,
            DHW, HW,
        )
        return out