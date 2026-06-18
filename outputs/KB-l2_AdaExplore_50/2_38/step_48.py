import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused AvgPool3d (kernel_size=2) kernel
# Input:  (B, C, D, H, W)
# Output: (B, C, D/2, H/2, W/2)
# ---------------------------------------------------------------------------
@triton.jit
def _avgpool3d_k2_kernel(
    x_ptr, out_ptr,
    B, C, D, H, W,
    Do, Ho, Wo,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    bc = tl.program_id(1)  # b*C + c
    out_spatial = Do * Ho * Wo
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < out_spatial

    wo = offs % Wo
    ho = (offs // Wo) % Ho
    do = offs // (Wo * Ho)

    di = do * 2
    hi = ho * 2
    wi = wo * 2

    in_stride_c = D * H * W
    base = bc * in_stride_c

    # 8 loads
    o000 = base + (di + 0) * H * W + (hi + 0) * W + (wi + 0)
    o001 = base + (di + 0) * H * W + (hi + 0) * W + (wi + 1)
    o010 = base + (di + 0) * H * W + (hi + 1) * W + (wi + 0)
    o011 = base + (di + 0) * H * W + (hi + 1) * W + (wi + 1)
    o100 = base + (di + 1) * H * W + (hi + 0) * W + (wi + 0)
    o101 = base + (di + 1) * H * W + (hi + 0) * W + (wi + 1)
    o110 = base + (di + 1) * H * W + (hi + 1) * W + (wi + 0)
    o111 = base + (di + 1) * H * W + (hi + 1) * W + (wi + 1)

    v000 = tl.load(x_ptr + o000, mask=mask, other=0.0)
    v001 = tl.load(x_ptr + o001, mask=mask, other=0.0)
    v010 = tl.load(x_ptr + o010, mask=mask, other=0.0)
    v011 = tl.load(x_ptr + o011, mask=mask, other=0.0)
    v100 = tl.load(x_ptr + o100, mask=mask, other=0.0)
    v101 = tl.load(x_ptr + o101, mask=mask, other=0.0)
    v110 = tl.load(x_ptr + o110, mask=mask, other=0.0)
    v111 = tl.load(x_ptr + o111, mask=mask, other=0.0)

    s = (v000 + v001 + v010 + v011 + v100 + v101 + v110 + v111) * 0.125

    out_off = bc * out_spatial + offs
    tl.store(out_ptr + out_off, s, mask=mask)


def avgpool3d_k2(x):
    B, C, D, H, W = x.shape
    Do, Ho, Wo = D // 2, H // 2, W // 2
    out = torch.empty((B, C, Do, Ho, Wo), device=x.device, dtype=x.dtype)
    spatial = Do * Ho * Wo
    BLOCK = 256
    grid = (triton.cdiv(spatial, BLOCK), B * C)
    _avgpool3d_k2_kernel[grid](
        x, out,
        B, C, D, H, W,
        Do, Ho, Wo,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# Fused Clamp + Softmax(dim=spatial) + Scale kernel
# ---------------------------------------------------------------------------
@triton.jit
def _clamp_softmax_scale_kernel(
    x_ptr, scale_ptr, out_ptr,
    S,
    CLAMP_MIN: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    C = tl.num_programs(1)
    row_start = (b * C + c) * S

    scale_val = tl.load(scale_ptr + c)

    max_val = -float('inf')
    sum_val = 0.0
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=-float('inf'))
        v = tl.minimum(tl.maximum(v, CLAMP_MIN), CLAMP_MAX)
        v = tl.where(mask, v, -float('inf'))
        block_max = tl.max(v, axis=0)
        new_max = tl.maximum(max_val, block_max)
        sum_val = sum_val * tl.exp(max_val - new_max)
        e = tl.exp(v - new_max)
        e = tl.where(mask, e, 0.0)
        sum_val += tl.sum(e, axis=0)
        max_val = new_max

    inv_sum = 1.0 / sum_val

    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        v = tl.minimum(tl.maximum(v, CLAMP_MIN), CLAMP_MAX)
        e = tl.exp(v - max_val) * inv_sum * scale_val
        tl.store(out_ptr + row_start + idx, e, mask=mask)


def fused_clamp_softmax_scale(x, scale, clamp_min, clamp_max):
    B, C, D, H, W = x.shape
    S = D * H * W
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    scale_flat = scale.contiguous().view(-1)

    if S <= 256:
        BLOCK = 256
        num_warps = 4
    elif S <= 1024:
        BLOCK = 1024
        num_warps = 4
    elif S <= 4096:
        BLOCK = 2048
        num_warps = 8
    else:
        BLOCK = 4096
        num_warps = 8

    grid = (B, C)
    _clamp_softmax_scale_kernel[grid](
        x_c, scale_flat, out,
        S,
        float(clamp_min), float(clamp_max),
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.pool_kernel_size = pool_kernel_size
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    def forward(self, x):
        # Custom avgpool for the common case of kernel_size=2 and aligned shapes
        if (self.pool_kernel_size == 2
                and x.is_cuda and x.dtype == torch.float32
                and x.shape[2] % 2 == 0 and x.shape[3] % 2 == 0 and x.shape[4] % 2 == 0):
            x = avgpool3d_k2(x.contiguous())
        else:
            x = self.avg_pool(x)
        x = self.conv_transpose(x)
        x = fused_clamp_softmax_scale(x, self.scale, self.clamp_min, self.clamp_max)
        return x