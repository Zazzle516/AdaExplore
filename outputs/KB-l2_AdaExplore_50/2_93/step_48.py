import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
    ],
    key=['N', 'OC', 'OH', 'OW', 'IC'],
)
@triton.jit
def conv_transpose_fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr,
    add_value: tl.constexpr,
    multiply_value: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # pid_m: tile over (OH*OW), pid_n: tile over OC, pid_b: batch
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    # Output spatial offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M] over OH*OW
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N] over OC

    oh = offs_m // OW   # [BLOCK_M]
    ow = offs_m % OW    # [BLOCK_M]

    m_mask = offs_m < (OH * OW)
    n_mask = offs_n < OC

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # For each (kh, kw), determine the corresponding input position.
    # ConvTranspose2d (no padding): out[oh, ow] += sum over (ic, kh, kw) where
    #   ih = (oh - kh) / stride, iw = (ow - kw) / stride
    # and (oh - kh) % stride == 0, ih in [0, IH).
    for kh in tl.static_range(0, KH):
        ih_num = oh - kh   # [BLOCK_M]
        ih_div = ih_num // STRIDE
        ih_mod = ih_num - ih_div * STRIDE
        ih_valid = (ih_mod == 0) & (ih_div >= 0) & (ih_div < IH)
        for kw in tl.static_range(0, KW):
            iw_num = ow - kw
            iw_div = iw_num // STRIDE
            iw_mod = iw_num - iw_div * STRIDE
            iw_valid = (iw_mod == 0) & (iw_div >= 0) & (iw_div < IW)
            valid = ih_valid & iw_valid & m_mask  # [BLOCK_M]

            # Load input block for all IC at this (ih_div, iw_div) for current batch
            # input shape: [N, IC, IH, IW], stride = (IC*IH*IW, IH*IW, IW, 1)
            # For each m in BLOCK_M, we need x[pid_b, :, ih_div, iw_div] for all IC
            # We'll loop over IC in chunks
            ih_safe = tl.where(valid, ih_div, 0)
            iw_safe = tl.where(valid, iw_div, 0)
            # base input offset per m
            x_spatial_off = ih_safe * IW + iw_safe  # [BLOCK_M]
            x_batch_off = pid_b * IC * IH * IW

            # weight shape [IC, OC, KH, KW], stride = (OC*KH*KW, KH*KW, KW, 1)
            # w[ic, oc, kh, kw] for oc in offs_n
            w_kk_off = kh * KW + kw  # scalar

            # Loop over IC
            BLOCK_K: tl.constexpr = 16
            for ic_start in range(0, IC, BLOCK_K):
                ic_offs = ic_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
                ic_mask = ic_offs < IC

                # Load x[pid_b, ic, ih, iw] -> shape [BLOCK_M, BLOCK_K]
                x_ptrs = x_ptr + x_batch_off + ic_offs[None, :] * (IH * IW) + x_spatial_off[:, None]
                x_load_mask = valid[:, None] & ic_mask[None, :]
                x_vals = tl.load(x_ptrs, mask=x_load_mask, other=0.0)

                # Load w[ic, oc, kh, kw] -> shape [BLOCK_K, BLOCK_N]
                w_ptrs = w_ptr + ic_offs[:, None] * (OC * KH * KW) + offs_n[None, :] * (KH * KW) + w_kk_off
                w_load_mask = ic_mask[:, None] & n_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_load_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals)

    # Add bias (folded with add_value)
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)  # [BLOCK_N]
    acc = acc + bias[None, :] + add_value

    # min(x, 0)
    acc = tl.minimum(acc, 0.0)

    # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.70710678118654752440
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # multiply
    acc = acc * multiply_value

    # Store output: out[pid_b, oc, oh, ow]
    # shape [N, OC, OH, OW]
    out_off = (pid_b * OC * OH * OW
               + offs_n[None, :] * (OH * OW)
               + offs_m[:, None])
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


def fused_conv_transpose(x, weight, bias, stride, add_value, multiply_value):
    N, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w
    OH = (IH - 1) * stride + KH
    OW = (IW - 1) * stride + KW

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(OH * OW, META['BLOCK_M']),
        triton.cdiv(OC, META['BLOCK_N']),
        N,
    )

    conv_transpose_fused_kernel[grid](
        x, weight, bias, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        stride,
        float(add_value),
        float(multiply_value),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        return fused_conv_transpose(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.stride,
            self.add_value,
            self.multiply_value,
        )