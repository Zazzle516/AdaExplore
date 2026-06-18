import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 256}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 8192}, num_warps=8, num_stages=2),
    ],
    key=['S', 'CPG'],
)
@triton.jit
def swish_groupnorm_hardswish_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    N, C, S,
    G, CPG: tl.constexpr,
    eps,
    BLOCK_S: tl.constexpr,
):
    # one program per (n, g)
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    group_size = CPG * S
    base = n * C * S + g * CPG * S

    offs_c = tl.arange(0, CPG)
    offs_s = tl.arange(0, BLOCK_S)

    num_s_blocks = (S + BLOCK_S - 1) // BLOCK_S

    sum_val = tl.zeros([], dtype=tl.float32)
    sum_sq = tl.zeros([], dtype=tl.float32)

    for sb in range(0, num_s_blocks):
        s_idx = sb * BLOCK_S + offs_s
        s_mask = s_idx < S
        mask = s_mask[None, :]
        ptrs = base + offs_c[:, None] * S + s_idx[None, :]
        x = tl.load(x_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
        sw = x * tl.sigmoid(x)
        sw = tl.where(mask, sw, 0.0)
        sum_val += tl.sum(sw)
        sum_sq += tl.sum(sw * sw)

    mean = sum_val / group_size
    var = sum_sq / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    gc_idx = g * CPG + offs_c
    w = tl.load(weight_ptr + gc_idx).to(tl.float32)
    b = tl.load(bias_ptr + gc_idx).to(tl.float32)

    for sb in range(0, num_s_blocks):
        s_idx = sb * BLOCK_S + offs_s
        s_mask = s_idx < S
        mask = s_mask[None, :]
        ptrs = base + offs_c[:, None] * S + s_idx[None, :]
        x = tl.load(x_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
        sw = x * tl.sigmoid(x)
        norm = (sw - mean) * rstd
        out = norm * w[:, None] + b[:, None]
        t = out + 3.0
        t = tl.minimum(tl.maximum(t, 0.0), 6.0)
        res = out * t * (1.0 / 6.0)
        tl.store(y_ptr + ptrs, res, mask=mask)


def fused_swish_gn_hswish(x, weight, bias, G, eps):
    # x: (N, C, D, H, W) contiguous
    N, C, D, H, W = x.shape
    S = D * H * W
    CPG = C // G
    x_flat = x.contiguous().view(N, C, S)
    y = torch.empty_like(x_flat)

    grid = (N * G,)
    swish_groupnorm_hardswish_kernel[grid](
        x_flat, y, weight, bias,
        N, C, S, G, CPG,
        eps,
    )
    return y.view(N, C, D, H, W)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.groups = groups
        self.eps = eps
        self.out_channels = out_channels
        # try channels_last_3d for the conv weights
        try:
            self.conv_transpose = self.conv_transpose.to(memory_format=torch.channels_last_3d)
        except Exception:
            pass

    def forward(self, x):
        try:
            x = x.contiguous(memory_format=torch.channels_last_3d)
        except Exception:
            pass
        x = self.conv_transpose(x)
        x = fused_swish_gn_hswish(x, self.group_norm.weight, self.group_norm.bias, self.groups, self.eps)
        return x