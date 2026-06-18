import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=8, num_stages=2),
    ],
    key=['total'],
)
@triton.jit
def fused_epilogue_kernel(
    x_ptr, sum_ptr, bias_ptr, out_ptr,
    total,
    SPATIAL: tl.constexpr,
    C_MASK: tl.constexpr,
    LOG2_SPATIAL: tl.constexpr,
    NEG_SLOPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    c_idx = (offs >> LOG2_SPATIAL) & C_MASK

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    s = tl.load(sum_ptr + c_idx, mask=mask, other=0.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    # add conv bias
    x = x + b
    # leaky relu
    x = tl.where(x >= 0, x, x * NEG_SLOPE)
    # add sum tensor
    x = x + s
    # clamp [-1, 1]
    x = tl.maximum(x, -1.0)
    x = tl.minimum(x, 1.0)
    # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.70710678118654752440
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(out_ptr + offs, x, mask=mask)


@triton.jit
def fused_epilogue_kernel_generic(
    x_ptr, sum_ptr, bias_ptr, out_ptr,
    total, spatial, C,
    NEG_SLOPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    c_idx = (offs // spatial) % C

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    s = tl.load(sum_ptr + c_idx, mask=mask, other=0.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    x = x + b
    x = tl.where(x >= 0, x, x * NEG_SLOPE)
    x = x + s
    x = tl.maximum(x, -1.0)
    x = tl.minimum(x, 1.0)
    inv_sqrt2 = 0.70710678118654752440
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(out_ptr + offs, x, mask=mask)


def _is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def fused_epilogue(x, sum_tensor, bias, neg_slope=0.2):
    x = x.contiguous()
    sum_flat = sum_tensor.contiguous().view(-1)
    bias_flat = bias.contiguous().view(-1)
    out = x  # in-place
    N, C, D, H, W = x.shape
    spatial = D * H * W
    total = x.numel()

    if _is_pow2(C) and _is_pow2(spatial):
        log2_spatial = int(math.log2(spatial))
        c_mask = C - 1
        grid = lambda meta: ((total + meta['BLOCK'] - 1) // meta['BLOCK'],)
        fused_epilogue_kernel[grid](
            x, sum_flat, bias_flat, out,
            total,
            SPATIAL=spatial,
            C_MASK=c_mask,
            LOG2_SPATIAL=log2_spatial,
            NEG_SLOPE=neg_slope,
        )
    else:
        BLOCK = 2048
        grid = ((total + BLOCK - 1) // BLOCK,)
        fused_epilogue_kernel_generic[grid](
            x, sum_flat, bias_flat, out,
            total, spatial, C,
            NEG_SLOPE=neg_slope,
            BLOCK=BLOCK,
            num_warps=8,
        )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super().__init__()
        conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        # Move bias out of conv; we'll fuse it into the epilogue.
        self.bias = nn.Parameter(conv.bias.detach().clone())
        conv.bias = None
        self.conv = conv
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x):
        x = self.conv(x)
        x = fused_epilogue(x, self.sum_tensor, self.bias, neg_slope=0.2)
        return x