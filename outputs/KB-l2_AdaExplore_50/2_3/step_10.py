import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_ln_pool_gelu_kernel(
    x_ptr,           # input: [N, C, D, H, W]
    w_ptr,           # LN weight [C]
    b_ptr,           # LN bias [C]
    out_ptr,         # output: [N, C, D//2, H//2, W//2]
    sum_weight,      # scalar to add
    N, C, D, H, W,
    Do, Ho, Wo,
    eps,
    BLOCK_C: tl.constexpr,
):
    # one program per output voxel (n, do, ho, wo)
    pid = tl.program_id(0)
    # pid layout: n * (Do*Ho*Wo) + do*Ho*Wo + ho*Wo + wo
    wo = pid % Wo
    tmp = pid // Wo
    ho = tmp % Ho
    tmp = tmp // Ho
    do = tmp % Do
    n = tmp // Do

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    # 8 input voxels per output (2x2x2 pool)
    # base index for input
    d0 = do * 2
    h0 = ho * 2
    w0 = wo * 2

    # strides for input [N, C, D, H, W]
    stride_n = C * D * H * W
    stride_c = D * H * W
    stride_d = H * W
    stride_h = W

    # accumulator for sum over 8 voxels of normalized values
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    w_vals = tl.load(w_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    b_vals = tl.load(b_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)

    for dd in tl.static_range(0, 2):
        for hh in tl.static_range(0, 2):
            for ww in tl.static_range(0, 2):
                d_idx = d0 + dd
                h_idx = h0 + hh
                w_idx = w0 + ww
                base = n * stride_n + d_idx * stride_d + h_idx * stride_h + w_idx
                ptrs = x_ptr + base + c_offs * stride_c
                vals = tl.load(ptrs, mask=c_mask, other=0.0).to(tl.float32)
                vals = vals + sum_weight
                # compute mean over channel
                # use masked sum
                vals_masked = tl.where(c_mask, vals, 0.0)
                mean = tl.sum(vals_masked, axis=0) / C
                diff = tl.where(c_mask, vals - mean, 0.0)
                var = tl.sum(diff * diff, axis=0) / C
                rstd = 1.0 / tl.sqrt(var + eps)
                normed = (vals - mean) * rstd * w_vals + b_vals
                acc += normed

    acc = acc / 8.0
    # GELU (exact via erf)
    inv_sqrt2 = 0.70710678118654752440
    gelu = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # write output [N, C, Do, Ho, Wo]
    out_stride_n = C * Do * Ho * Wo
    out_stride_c = Do * Ho * Wo
    out_stride_d = Ho * Wo
    out_stride_h = Wo
    out_base = n * out_stride_n + do * out_stride_d + ho * out_stride_h + wo
    out_ptrs = out_ptr + out_base + c_offs * out_stride_c
    tl.store(out_ptrs, gelu, mask=c_mask)


def fused_ln_pool_gelu(x, weight, bias, sum_weight, eps):
    N, C, D, H, W = x.shape
    Do, Ho, Wo = D // 2, H // 2, W // 2
    out = torch.empty((N, C, Do, Ho, Wo), device=x.device, dtype=x.dtype)
    # BLOCK_C must be power of 2 >= C
    BLOCK_C = 64
    while BLOCK_C < C:
        BLOCK_C *= 2
    grid = (N * Do * Ho * Wo,)
    fused_ln_pool_gelu_kernel[grid](
        x, weight, bias, out,
        float(sum_weight),
        N, C, D, H, W,
        Do, Ho, Wo,
        eps,
        BLOCK_C=BLOCK_C,
        num_warps=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = self.conv_transpose(x)
        # Check if we can use fused kernel: pool 2x2x2, norm over channels
        if (self.pool_kernel_size == (2, 2, 2) and
            x.shape[2] % 2 == 0 and x.shape[3] % 2 == 0 and x.shape[4] % 2 == 0):
            x = x.contiguous()
            return fused_ln_pool_gelu(
                x, self.norm.weight, self.norm.bias,
                self.sum_weight.item(), self.norm.eps
            )
        else:
            x = x + self.sum_weight
            x = self.norm(x)
            x = self.avg_pool(x)
            x = self.gelu(x)
            return x