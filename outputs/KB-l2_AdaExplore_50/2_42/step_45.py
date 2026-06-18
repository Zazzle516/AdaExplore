import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# We must perform the full ConvTranspose2d work (per safety contract).
# However, after conv_transpose we do global average pooling over (H_out, W_out).
# We still must materialize the full output shape and asymptotic FLOPs of conv_transpose.
# Strategy: implement conv_transpose2d as a direct kernel that computes each output
# pixel and accumulates into a per-(N, OC) sum buffer. We never write the full
# H_out*W_out output to global memory, but we do all the multiply-adds.
#
# Actually, we need to materialize the output shape. Let's just use torch's
# conv_transpose2d to be safe, then write custom kernels for the rest.

@triton.jit
def mean_bias_lse_sum_kernel(
    x_ptr,        # [N, C, H, W]
    bias_ptr,     # [C]
    out_ptr,      # [N]
    N, C, H, W,
    scale,        # 10.0 / (H*W) ... wait, we need scale separately
    BLOCK_C: tl.constexpr,
):
    # one program per batch sample
    n = tl.program_id(0)
    
    HW = H * W
    inv_hw = 1.0 / HW.to(tl.float32)
    
    # Compute per-channel mean: mean[c] = sum over h,w of x[n,c,h,w] / HW
    # Then add bias: v[c] = mean[c] + bias[c]
    # Then logsumexp over c: lse = log(sum(exp(v[c] - max_v))) + max_v
    # Then sum over (1,1) is just lse, multiply by 10.
    
    # First pass: compute mean per channel, find max
    offs_c = tl.arange(0, BLOCK_C)
    
    # We need to compute mean per channel. We'll loop channels in chunks.
    # For each channel, sum all H*W elements.
    
    # Simpler: process channels in blocks of BLOCK_C, but compute their sums.
    # Since H*W is large (e.g. 514*514), we loop spatially.
    
    # Actually we'll do: for each channel c, accumulate sum.
    # Use a 2D tile: BLOCK_C channels x BLOCK_HW spatial.
    pass


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 8192}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_HW': 16384}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 16384}, num_warps=16, num_stages=2),
    ],
    key=['C', 'H', 'W', 'SPLITS'],
)
@triton.jit
def partial_sum_kernel(
    x_ptr,        # [N, C, H, W]
    partial_ptr,  # [N, C, SPLITS] float32
    N, C, H, W,
    SPLITS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    s = tl.program_id(2)
    
    HW = H * W
    chunk = (HW + SPLITS - 1) // SPLITS
    start = s * chunk
    end = start + chunk
    if end > HW:
        end = HW
    
    base = n * C * HW + c * HW
    
    acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)
    offs = tl.arange(0, BLOCK_HW)
    for off in range(start, end, BLOCK_HW):
        idx = off + offs
        mask = idx < end
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        acc += tl.where(mask, v.to(tl.float32), 0.0)
    
    total = tl.sum(acc, axis=0)
    tl.store(partial_ptr + n * C * SPLITS + c * SPLITS + s, total)


@triton.jit
def finalize_mean_bias_kernel(
    partial_ptr,  # [N, C, SPLITS]
    bias_ptr,     # [C]
    inter_ptr,    # [N, C]
    N, C,
    HW,
    SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    
    offs = tl.arange(0, BLOCK_S)
    mask = offs < SPLITS
    v = tl.load(partial_ptr + n * C * SPLITS + c * SPLITS + offs, mask=mask, other=0.0)
    s = tl.sum(v, axis=0)
    mean = s / HW.to(tl.float32)
    b = tl.load(bias_ptr + c).to(tl.float32)
    tl.store(inter_ptr + n * C + c, mean + b)


@triton.jit
def lse_kernel(
    inter_ptr,   # [N, C]
    out_ptr,     # [N]
    N, C,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    
    v = tl.load(inter_ptr + n * C + offs, mask=mask, other=-float('inf')).to(tl.float32)
    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    lse = tl.log(s) + m
    res = lse * 10.0
    tl.store(out_ptr + n, res)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.out_channels = out_channels

    def forward(self, x):
        # Full conv_transpose - do real work
        y = self.conv_transpose(x)  # [N, C, H, W]
        N, C, H, W = y.shape
        
        bias_flat = self.bias.view(-1).contiguous()
        y = y.contiguous()
        
        inter = torch.empty((N, C), device=y.device, dtype=torch.float32)
        out = torch.empty((N, 1), device=y.device, dtype=y.dtype)
        
        # Split spatial reduction across multiple programs for more parallelism
        SPLITS = 8
        BLOCK_S = 8
        partial = torch.empty((N, C, SPLITS), device=y.device, dtype=torch.float32)
        
        grid1 = (N, C, SPLITS)
        partial_sum_kernel[grid1](y, partial, N, C, H, W, SPLITS=SPLITS)
        
        grid_f = (N, C)
        finalize_mean_bias_kernel[grid_f](partial, bias_flat, inter, N, C, H * W,
                                          SPLITS=SPLITS, BLOCK_S=BLOCK_S,
                                          num_warps=1)
        
        # next power of 2 >= C
        BLOCK_C = 1
        while BLOCK_C < C:
            BLOCK_C *= 2
        
        grid2 = (N,)
        lse_kernel[grid2](inter, out, N, C, BLOCK_C=BLOCK_C)
        
        return out