import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    # program ids: (n, oc_tile, sp_tile)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (OH * OW)

    oh = sp_offs // OW
    ow = sp_offs % OW

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # Loop over input channels and kernel
    for ic in range(0, IC_C):
        ic_valid = ic < IC
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = oh + kh
                iw = ow + kw
                # x[n, ic, ih, iw]
                x_idx = pid_n * (IC * IH * IW) + ic * (IH * IW) + ih * IW + iw
                x_vals = tl.load(x_ptr + x_idx, mask=sp_mask & ic_valid, other=0.0)  # [BLOCK_SP]
                # w[oc, ic, kh, kw]
                w_idx = oc_offs * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                w_vals = tl.load(w_ptr + w_idx, mask=oc_mask & ic_valid, other=0.0)  # [BLOCK_OC]
                acc += w_vals[:, None] * x_vals[None, :]

    # Add bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += bias[:, None]

    # Store
    out_idx = pid_n * (OC * OH * OW) + oc_offs[:, None] * (OH * OW) + sp_offs[None, :]
    mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_idx, acc, mask=mask)


@triton.jit
def gn_scale_pool_clamp_kernel(
    x_ptr, gamma_ptr, beta_ptr, scale_ptr, out_ptr,
    N, C, H, W,
    OH: tl.constexpr, OW: tl.constexpr,
    G: tl.constexpr,
    CPG: tl.constexpr,
    POOL: tl.constexpr,
    HW: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    eps: tl.constexpr,
    clamp_min: tl.constexpr,
    clamp_max: tl.constexpr,
):
    # one program per (n, g)
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    # compute mean and var over GROUP_SIZE = CPG * H * W elements
    offs = tl.arange(0, BLOCK)
    base = pid_n * (C * H * W) + pid_g * (CPG * H * W)

    sum_x = tl.zeros((), dtype=tl.float32)
    sum_x2 = tl.zeros((), dtype=tl.float32)

    for start in range(0, GROUP_SIZE, BLOCK):
        idx = start + offs
        mask = idx < GROUP_SIZE
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        sum_x += tl.sum(v, axis=0)
        sum_x2 += tl.sum(v * v, axis=0)

    mean = sum_x / GROUP_SIZE
    var = sum_x2 / GROUP_SIZE - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    TOTAL_OUT = OH * OW
    out_offs = tl.arange(0, BLOCK_OUT)

    # Iterate channels in group
    for c_in_g in tl.static_range(0, CPG):
        c = pid_g * CPG + c_in_g
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)
        scale = tl.load(scale_ptr + c)
        a = rstd * gamma * scale
        b = beta * scale - mean * rstd * gamma * scale

        ch_base = pid_n * (C * H * W) + c * (H * W)
        out_base = pid_n * (C * OH * OW) + c * (OH * OW)

        for out_start in range(0, TOTAL_OUT, BLOCK_OUT):
            idx_out = out_start + out_offs
            mask_out = idx_out < TOTAL_OUT
            oh = idx_out // OW
            ow = idx_out % OW
            max_val = tl.full((BLOCK_OUT,), -1e30, dtype=tl.float32)
            for ph in tl.static_range(0, POOL):
                for pw in tl.static_range(0, POOL):
                    ih = oh * POOL + ph
                    iw = ow * POOL + pw
                    in_idx = ch_base + ih * W + iw
                    v = tl.load(x_ptr + in_idx, mask=mask_out, other=-1e30)
                    y = v * a + b
                    max_val = tl.maximum(max_val, y)
            max_val = tl.minimum(tl.maximum(max_val, clamp_min), clamp_max)
            tl.store(out_ptr + out_base + idx_out, max_val, mask=mask_out)


def triton_conv2d(x, w, b):
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = w.shape
    OH = IH - KH + 1
    OW = IW - KW + 1
    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 32
    BLOCK_SP = 64
    # IC_C: round up IC to a compile-time constant
    IC_C = IC  # IC is small (8)

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_SP))
    conv_kernel[grid](
        x, w, b, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        IC_C,
        BLOCK_OC, BLOCK_SP,
        num_warps=4, num_stages=2,
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
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.maxpool_kernel_size = maxpool_kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()

        # Custom conv
        y = triton_conv2d(x, w, b)

        N, C, H, W = y.shape
        G = self.num_groups
        CPG = C // G
        POOL = self.maxpool_kernel_size
        OH = H // POOL
        OW = W // POOL

        out = torch.empty((N, C, OH, OW), device=y.device, dtype=y.dtype)

        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()
        scale = self.scale.contiguous().view(-1)
        eps = self.group_norm.eps

        GROUP_SIZE = CPG * H * W
        # pick BLOCK
        BLOCK = 1024
        while BLOCK > GROUP_SIZE:
            BLOCK //= 2
        if BLOCK < 64:
            BLOCK = 64

        BLOCK_OUT = 128
        grid = (N, G)
        gn_scale_pool_clamp_kernel[grid](
            y, gamma, beta, scale, out,
            N, C, H, W,
            OH, OW,
            G, CPG, POOL,
            H * W,
            GROUP_SIZE,
            BLOCK,
            BLOCK_OUT,
            eps,
            float(self.clamp_min),
            float(self.clamp_max),
            num_warps=8, num_stages=2,
        )
        return out