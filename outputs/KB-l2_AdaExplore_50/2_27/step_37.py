import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=3),
    ],
    key=['C', 'S'],
)
@triton.jit
def fused_post_conv_kernel(
    x_ptr,
    out_ptr,
    gamma_ptr,
    beta_ptr,
    B, C, S,
    inv_s, inv_n,
    eps: tl.constexpr,
    BLOCK_S: tl.constexpr,
    CPG: tl.constexpr,
):
    b = tl.program_id(0)
    g = tl.program_id(1)

    c_offs = tl.arange(0, CPG) + g * CPG
    s_offs = tl.arange(0, BLOCK_S)

    base = b * (C * S)
    mask_s = s_offs < S
    ptrs = x_ptr + base + c_offs[:, None] * S + s_offs[None, :]
    mask = mask_s[None, :]
    x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)

    t = x + 3.0
    t = tl.maximum(t, 0.0)
    t = tl.minimum(t, 6.0)
    hs = x * t * (1.0 / 6.0)
    hs = tl.where(mask, hs, 0.0)

    ch_sum = tl.sum(hs, axis=1)
    sum_val = tl.sum(ch_sum)
    sumsq_val = tl.sum(hs * hs)

    mean = sum_val * inv_n
    var = sumsq_val * inv_n - mean * mean
    rstd = tl.rsqrt(var + eps)

    gamma = tl.load(gamma_ptr + c_offs).to(tl.float32)
    beta = tl.load(beta_ptr + c_offs).to(tl.float32)
    out_val = (ch_sum * inv_s - mean) * rstd * gamma + beta
    tl.store(out_ptr + b * C + c_offs, out_val)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv(x)
        B, C, D, H, W = x.shape
        S = D * H * W
        x_flat = x.contiguous().view(B, C, S)
        out = torch.empty((B, C), device=x.device, dtype=x.dtype)

        cpg = C // self.num_groups
        grid = (B, self.num_groups)
        inv_s = 1.0 / float(S)
        inv_n = 1.0 / float(S * cpg)
        # Next power of two >= S
        block_s = 1
        while block_s < S:
            block_s *= 2
        fused_post_conv_kernel[grid](
            x_flat, out,
            self.group_norm.weight, self.group_norm.bias,
            B, C, S,
            inv_s, inv_n,
            eps=self.eps,
            BLOCK_S=block_s,
            CPG=cpg,
        )
        return out