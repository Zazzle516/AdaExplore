import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    oh = sp_offs // OW
    ow = sp_offs % OW

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < OH * OW

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    for ic in range(0, IC):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = oh + kh
                iw = ow + kw
                x_offs = pid_n * IC * IH * IW + ic * IH * IW + ih * IW + iw
                x_vals = tl.load(x_ptr + x_offs, mask=sp_mask, other=0.0)  # [BLOCK_SP]
                w_offs = oc_offs * (IC * KH * KW) + ic * KH * KW + kh * KW + kw
                w_vals = tl.load(w_ptr + w_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                acc += w_vals[:, None] * x_vals[None, :]

    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b_vals[:, None]

    out_offs = pid_n * OC * OH * OW + oc_offs[:, None] * OH * OW + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


def triton_conv2d(x, w, b):
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = w.shape
    OH = IH - KH + 1
    OW = IW - KW + 1
    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)
    BLOCK_OC = 32
    BLOCK_SP = 128
    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_SP))
    conv2d_kernel[grid](
        x, w, b, out,
        N, IC, IH, IW, OC, OH, OW,
        KH, KW, BLOCK_OC, BLOCK_SP,
    )
    return out


@triton.jit
def gn_scale_maxpool_clamp_kernel(
    x_ptr,           # [N, C, H, W] after conv
    gamma_ptr,       # [C] group_norm.weight
    beta_ptr,        # [C] group_norm.bias
    scale_ptr,       # [C] scale (flattened)
    out_ptr,         # [N, C, OH, OW]
    N, C, H, W,
    OH, OW,
    G: tl.constexpr,
    C_PER_G: tl.constexpr,
    HW: tl.constexpr,
    GROUP_SIZE: tl.constexpr,  # C_PER_G * HW
    POOL: tl.constexpr,
    eps: tl.constexpr,
    clamp_min: tl.constexpr,
    clamp_max: tl.constexpr,
    BLOCK: tl.constexpr,       # power-of-two >= GROUP_SIZE
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    base = n * C * H * W + g * C_PER_G * H * W

    offs = tl.arange(0, BLOCK)
    mask = offs < GROUP_SIZE

    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)

    # compute mean/var over group
    sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
    sum_x2 = tl.sum(tl.where(mask, x * x, 0.0), axis=0)
    mean = sum_x / GROUP_SIZE
    var = sum_x2 / GROUP_SIZE - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Now we need per-channel gamma, beta, scale.
    # For maxpool, iterate over channels in this group, and over output positions.
    # We'll process channel by channel for clarity.
    for c_local in tl.static_range(0, C_PER_G):
        c = g * C_PER_G + c_local
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)
        sc = tl.load(scale_ptr + c)

        # For each output pool location, compute max over the pool window
        # Output spatial size OH*OW. Use a flat range.
        out_offs = tl.arange(0, BLOCK)  # reuse BLOCK; need BLOCK >= OH*OW
        out_mask = out_offs < (OH * OW)
        oh_idx = out_offs // OW
        ow_idx = out_offs % OW

        # max over pool window
        max_val = tl.full((BLOCK,), -float('inf'), dtype=tl.float32)
        for ph in tl.static_range(0, POOL):
            for pw in tl.static_range(0, POOL):
                ih = oh_idx * POOL + ph
                iw = ow_idx * POOL + pw
                # load from x corresponding to channel c_local, spatial (ih, iw)
                # within the group: offset = c_local * HW + ih * W + iw
                idx_in_group = c_local * HW + ih * W + iw
                in_mask = out_mask & (ih < H) & (iw < W)
                v = tl.load(x_ptr + base + idx_in_group, mask=in_mask, other=-float('inf'))
                # normalize, affine, scale
                v_n = (v - mean) * rstd
                v_n = v_n * gamma + beta
                v_n = v_n * sc
                max_val = tl.where(v_n > max_val, v_n, max_val)

        # clamp
        out_v = tl.minimum(tl.maximum(max_val, clamp_min), clamp_max)
        out_base = n * C * OH * OW + c * OH * OW
        tl.store(out_ptr + out_base + out_offs, out_v, mask=out_mask)


def triton_gn_scale_maxpool_clamp(x, gamma, beta, scale, num_groups, pool, clamp_min, clamp_max, eps=1e-5):
    N, C, H, W = x.shape
    C_per_G = C // num_groups
    OH = H // pool
    OW = W // pool
    HW = H * W
    GROUP_SIZE = C_per_G * HW
    # BLOCK must be >= max(GROUP_SIZE, OH*OW), power of two
    needed = max(GROUP_SIZE, OH * OW)
    BLOCK = 1
    while BLOCK < needed:
        BLOCK *= 2

    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    grid = (N * num_groups,)
    gn_scale_maxpool_clamp_kernel[grid](
        x, gamma, beta, scale.contiguous().view(-1), out,
        N, C, H, W, OH, OW,
        num_groups, C_per_G, HW, GROUP_SIZE,
        pool, eps, clamp_min, clamp_max,
        BLOCK,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.num_groups = num_groups
        self.maxpool_kernel_size = maxpool_kernel_size

    def forward(self, x):
        x = x.cuda().contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        # conv
        y = triton_conv2d(x, w, b)
        # group norm + scale + maxpool + clamp fused
        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()
        scale = self.scale.contiguous()
        out = triton_gn_scale_maxpool_clamp(
            y, gamma, beta, scale,
            self.num_groups, self.maxpool_kernel_size,
            float(self.clamp_min), float(self.clamp_max),
            float(self.group_norm.eps),
        )
        return out