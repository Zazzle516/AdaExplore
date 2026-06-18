import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'M_TOTAL', 'K_TOTAL'],
)
@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, b_ptr, bias_ptr, out_ptr,
    N, IC, D, H, W,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    M_TOTAL, K_TOTAL,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wd, stride_wh, stride_ww,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # spatial+batch tile (M)
    pid_n = tl.program_id(1)  # OC tile (N)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Decode m -> (n, od, oh, ow)
    m_mask = offs_m < M_TOTAL
    OHW = OH * OW
    ODHW = OD * OHW
    n_idx = offs_m // ODHW
    rem = offs_m % ODHW
    od_idx = rem // OHW
    rem2 = rem % OHW
    oh_idx = rem2 // OW
    ow_idx = rem2 % OW

    # K dimension = IC * KD * KH * KW
    KDHW = KD * KH * KW
    KHKW = KH * KW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K_TOTAL, BLOCK_K):
        k_idx = k_start + offs_k
        k_mask = k_idx < K_TOTAL

        ic = k_idx // KDHW
        krem = k_idx % KDHW
        kd = krem // KHKW
        krem2 = krem % KHKW
        kh = krem2 // KW
        kw = krem2 % KW

        # x index: [n, ic, od+kd, oh+kh, ow+kw]
        x_d = od_idx[:, None] + kd[None, :]
        x_h = oh_idx[:, None] + kh[None, :]
        x_w = ow_idx[:, None] + kw[None, :]
        x_offs = (n_idx[:, None] * stride_xn
                  + ic[None, :] * stride_xc
                  + x_d * stride_xd
                  + x_h * stride_xh
                  + x_w * stride_xw)
        x_mask = m_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

        # w index: [oc, ic, kd, kh, kw]
        w_offs = (offs_n[None, :] * stride_wo
                  + ic[:, None] * stride_wi
                  + kd[:, None] * stride_wd
                  + kh[:, None] * stride_wh
                  + kw[:, None] * stride_ww)
        w_mask = k_mask[:, None] & (offs_n[None, :] < OC)
        w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    # Add conv bias
    cb = tl.load(b_ptr + offs_n, mask=offs_n < OC, other=0.0)
    acc = acc + cb[None, :]

    # Fused activations: ReLU -> LeakyReLU(0.01) -> GELU -> Sigmoid -> +bias
    y = tl.maximum(acc, 0.0)
    # leaky_relu on non-negative is identity; skip
    inv_sqrt2 = 0.70710678118654752440
    y = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))
    y = tl.sigmoid(y)

    # extra bias is per-OC
    eb = tl.load(bias_ptr + offs_n, mask=offs_n < OC, other=0.0)
    y = y + eb[None, :]

    # Store
    out_offs = (n_idx[:, None] * stride_on
                + offs_n[None, :] * stride_oc
                + od_idx[:, None] * stride_od
                + oh_idx[:, None] * stride_oh
                + ow_idx[:, None] * stride_ow)
    out_mask = m_mask[:, None] & (offs_n[None, :] < OC)
    tl.store(out_ptr + out_offs, y, mask=out_mask)


def conv3d_fused(x, weight, conv_bias, extra_bias):
    x = x.contiguous()
    weight = weight.contiguous()
    N, IC, D, H, W = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = D - KD + 1
    OH = H - KH + 1
    OW = W - KW + 1

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    M_TOTAL = N * OD * OH * OW
    K_TOTAL = IC * KD * KH * KW

    extra_bias_flat = extra_bias.contiguous().view(-1)
    conv_bias_flat = conv_bias.contiguous().view(-1)

    grid = lambda META: (
        triton.cdiv(M_TOTAL, META['BLOCK_M']),
        triton.cdiv(OC, META['BLOCK_N']),
    )

    conv3d_fused_kernel[grid](
        x, weight, conv_bias_flat, extra_bias_flat, out,
        N, IC, D, H, W,
        OC, OD, OH, OW,
        KD, KH, KW,
        M_TOTAL, K_TOTAL,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.cuda().contiguous()
        return conv3d_fused(x, self.conv.weight, self.conv.bias, self.bias)