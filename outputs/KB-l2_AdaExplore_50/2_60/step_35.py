import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def conv_transpose3d_swish(x, weight, bias, stride, padding):
    # Use cuDNN-backed conv_transpose3d (heavy op runs at runtime).
    # Swish is fused into the downstream GroupNorm kernel.
    return F.conv_transpose3d(x, weight, bias, stride=stride, padding=padding)


@triton.jit
def groupnorm_hardswish_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    N, C, S,
    G, CPG,
    eps,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    group_size = CPG * S
    base = n * C * S + g * CPG * S

    sum_val = tl.zeros([], dtype=tl.float32)
    sum_sq = tl.zeros([], dtype=tl.float32)

    offs_c = tl.arange(0, BLOCK_C)
    offs_s = tl.arange(0, BLOCK_S)

    num_s_blocks = (S + BLOCK_S - 1) // BLOCK_S

    for c_start in range(0, CPG, BLOCK_C):
        c_idx = c_start + offs_c
        c_mask = c_idx < CPG
        for sb in range(0, num_s_blocks):
            s_idx = sb * BLOCK_S + offs_s
            s_mask = s_idx < S
            mask = c_mask[:, None] & s_mask[None, :]
            ptrs = base + c_idx[:, None] * S + s_idx[None, :]
            raw = tl.load(x_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
            # Fused Swish
            x = raw * tl.sigmoid(raw)
            x = tl.where(mask, x, 0.0)
            sum_val += tl.sum(x)
            sum_sq += tl.sum(x * x)

    mean = sum_val / group_size
    var = sum_sq / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for c_start in range(0, CPG, BLOCK_C):
        c_idx = c_start + offs_c
        c_mask = c_idx < CPG
        gc_idx = g * CPG + c_idx
        w = tl.load(weight_ptr + gc_idx, mask=c_mask, other=0.0).to(tl.float32)
        b = tl.load(bias_ptr + gc_idx, mask=c_mask, other=0.0).to(tl.float32)
        for sb in range(0, num_s_blocks):
            s_idx = sb * BLOCK_S + offs_s
            s_mask = s_idx < S
            mask = c_mask[:, None] & s_mask[None, :]
            ptrs = base + c_idx[:, None] * S + s_idx[None, :]
            raw = tl.load(x_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
            x = raw * tl.sigmoid(raw)
            norm = (x - mean) * rstd
            out = norm * w[:, None] + b[:, None]
            t = out + 3.0
            t = tl.minimum(tl.maximum(t, 0.0), 6.0)
            res = out * t / 6.0
            tl.store(y_ptr + ptrs, res, mask=mask)


def fused_gn_hswish(x, weight, bias, G, eps):
    N, C, D, H, W = x.shape
    S = D * H * W
    CPG = C // G
    x_flat = x.contiguous().view(N, C, S)
    y = torch.empty_like(x_flat)

    BLOCK_S = 1024
    if S < 1024:
        BLOCK_S = 1
        while BLOCK_S < S:
            BLOCK_S *= 2
        BLOCK_S = max(BLOCK_S, 16)
    BLOCK_C = 4
    if CPG < 4:
        BLOCK_C = 1
        while BLOCK_C < CPG:
            BLOCK_C *= 2

    grid = (N * G,)
    groupnorm_hardswish_kernel[grid](
        x_flat, y, weight, bias,
        N, C, S, G, CPG,
        eps,
        BLOCK_S=BLOCK_S,
        BLOCK_C=BLOCK_C,
        num_warps=4,
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
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = x.contiguous()
        w = self.conv_transpose.weight
        b = self.conv_transpose.bias
        x = conv_transpose3d_swish(x, w, b, self.stride, self.padding)
        x = x.contiguous()
        x = fused_gn_hswish(x, self.group_norm.weight, self.group_norm.bias, self.groups, self.eps)
        return x