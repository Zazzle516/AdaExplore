import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_R': 1024, 'BLOCK_S': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_R': 2048, 'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_R': 2048, 'BLOCK_S': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_R': 4096, 'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_R': 1024, 'BLOCK_S': 2048}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_R': 512, 'BLOCK_S': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_R': 2048, 'BLOCK_S': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_R': 4096, 'BLOCK_S': 4096}, num_warps=8, num_stages=2),
    ],
    key=['group_elems', 'S', 'C_per_G'],
)
@triton.jit
def fused_swish_gn_hs_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    C, S, G, C_per_G: tl.constexpr, group_elems, eps,
    BLOCK_R: tl.constexpr, BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G
    base = n * C * S + g * C_per_G * S
    inv_group = 1.0 / group_elems

    # Pass 1: sum and sum_sq of swish(x)
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)
    offs_r = tl.arange(0, BLOCK_R)
    for start in range(0, group_elems, BLOCK_R):
        idx = start + offs_r
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        sw = x * tl.sigmoid(x)
        sw = tl.where(mask, sw, 0.0)
        sum_val += tl.sum(sw, axis=0)
        sum_sq += tl.sum(sw * sw, axis=0)

    mean = sum_val * inv_group
    var = sum_sq * inv_group - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2: 2D tile [C_per_G, BLOCK_S]; load weights/biases once per program
    g_base = g * C_per_G
    c_offs = tl.arange(0, C_per_G)
    w = tl.load(weight_ptr + g_base + c_offs).to(tl.float32)  # [C_per_G]
    b = tl.load(bias_ptr + g_base + c_offs).to(tl.float32)    # [C_per_G]

    spatial_offs = tl.arange(0, BLOCK_S)
    for s_start in range(0, S, BLOCK_S):
        s_idx = s_start + spatial_offs
        s_mask = s_idx < S
        # 2D index: (c_in_g, s_idx) -> base + c_in_g * S + s_idx
        ptrs = x_ptr + base + c_offs[:, None] * S + s_idx[None, :]
        m2 = s_mask[None, :]
        x = tl.load(ptrs, mask=m2, other=0.0).to(tl.float32)
        sw = x * tl.sigmoid(x)
        y = (sw - mean) * rstd * w[:, None] + b[:, None]
        t = y + 3.0
        t = tl.minimum(tl.maximum(t, 0.0), 6.0)
        out = y * t * (1.0 / 6.0)
        out_ptrs = out_ptr + base + c_offs[:, None] * S + s_idx[None, :]
        tl.store(out_ptrs, out, mask=m2)


def triton_fused_swish_gn_hs(x, weight, bias, G, eps):
    x = x.contiguous()
    N, C = x.shape[0], x.shape[1]
    S = 1
    for d in x.shape[2:]:
        S *= d
    C_per_G = C // G
    out = torch.empty_like(x)
    group_elems = C_per_G * S
    grid = (N * G,)
    fused_swish_gn_hs_kernel[grid](
        x, weight, bias, out,
        C, S, G, C_per_G, group_elems, eps,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.groups = groups
        self.eps = eps

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_fused_swish_gn_hs(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.groups,
            self.eps,
        )
        return x