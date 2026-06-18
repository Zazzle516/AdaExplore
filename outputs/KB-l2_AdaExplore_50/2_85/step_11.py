import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
    ],
    key=['C_OUT', 'H_OUT', 'W_OUT', 'C_IN'],
)
@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_IN, H_IN, W_IN,
    C_OUT, H_OUT, W_OUT,
    KH: tl.constexpr, KW: tl.constexpr,
    C_IN_C: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    oh = hw_offs // W_OUT
    ow = hw_offs % W_OUT

    oc_mask = oc_offs < C_OUT
    hw_mask = hw_offs < (H_OUT * W_OUT)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh
            iw = ow + kw
            for ic in tl.static_range(0, C_IN_C):
                w_idx = oc_offs * (C_IN_C * KH * KW) + ic * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)
                x_idx = pid_n * (C_IN_C * H_IN * W_IN) + ic * (H_IN * W_IN) + ih * W_IN + iw
                x_val = tl.load(x_ptr + x_idx, mask=hw_mask, other=0.0)
                acc += w_val[:, None] * x_val[None, :]

    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b_val[:, None]

    out_idx = pid_n * (C_OUT * H_OUT * W_OUT) + oc_offs[:, None] * (H_OUT * W_OUT) + hw_offs[None, :]
    out_mask = oc_mask[:, None] & hw_mask[None, :]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


def triton_conv2d(x, w, b):
    N, C_IN, H_IN, W_IN = x.shape
    C_OUT, _, KH, KW = w.shape
    H_OUT = H_IN - KH + 1
    W_OUT = W_IN - KW + 1
    out = torch.empty((N, C_OUT, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

    grid = lambda META: (N, triton.cdiv(C_OUT, META['BLOCK_OC']), triton.cdiv(H_OUT * W_OUT, META['BLOCK_HW']))
    conv2d_kernel[grid](
        x, w, b, out,
        N, C_IN, H_IN, W_IN,
        C_OUT, H_OUT, W_OUT,
        KH, KW, C_IN,
    )
    return out


@triton.jit
def fused_gn_pool_kernel(
    x_ptr, gamma_ptr, beta_ptr, scale_ptr, out_ptr,
    N, C, H, W, G, CPG: tl.constexpr,
    H_OUT, W_OUT,
    HW, GROUP_SIZE,
    POOL: tl.constexpr,
    CLAMP_MIN: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # one program per (n, g)
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    base = n * (C * HW) + g * CPG * HW

    # Pass 1: compute mean / rstd
    sum_ = tl.zeros((BLOCK,), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK,), dtype=tl.float32)

    for off in range(0, GROUP_SIZE, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < GROUP_SIZE
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        sum_ += tl.where(mask, v, 0.0)
        sum_sq += tl.where(mask, v * v, 0.0)

    s = tl.sum(sum_, axis=0)
    ssq = tl.sum(sum_sq, axis=0)
    mean = s / GROUP_SIZE
    var = ssq / GROUP_SIZE - mean * mean
    rstd = 1.0 / tl.sqrt(var + EPS)

    # Pass 2: for each channel in the group, do pool over its HW
    HW_OUT = H_OUT * W_OUT

    for c_in in tl.static_range(0, CPG):
        c = g * CPG + c_in
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)
        scale = tl.load(scale_ptr + c)
        a = rstd * gamma * scale
        bb = beta * scale - mean * rstd * gamma * scale

        ch_base = n * (C * HW) + c * HW
        out_base = n * (C * HW_OUT) + c * HW_OUT

        # iterate over output pixel tiles
        for ohw_start in range(0, HW_OUT, BLOCK):
            ohw = ohw_start + tl.arange(0, BLOCK)
            o_mask = ohw < HW_OUT
            oh = ohw // W_OUT
            ow = ohw % W_OUT

            acc = tl.full((BLOCK,), -float('inf'), dtype=tl.float32)
            for ph in tl.static_range(0, POOL):
                for pw in tl.static_range(0, POOL):
                    ih = oh * POOL + ph
                    iw = ow * POOL + pw
                    x_idx = ch_base + ih * W + iw
                    v = tl.load(x_ptr + x_idx, mask=o_mask, other=-float('inf'))
                    v_norm = v * a + bb
                    acc = tl.maximum(acc, v_norm)

            acc = tl.minimum(tl.maximum(acc, CLAMP_MIN), CLAMP_MAX)
            tl.store(out_ptr + out_base + ohw, acc, mask=o_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.num_groups = num_groups
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = 1e-5

    def forward(self, x):
        x = x.cuda().contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()

        # Conv
        y = triton_conv2d(x, w, b)
        N, C, H, W = y.shape

        G = self.num_groups
        CPG = C // G
        POOL = self.maxpool_kernel_size
        H_OUT = H // POOL
        W_OUT = W // POOL

        out = torch.empty((N, C, H_OUT, W_OUT), device=y.device, dtype=y.dtype)

        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()
        scale = self.scale.view(-1).contiguous()

        HW = H * W
        GROUP_SIZE = CPG * HW
        BLOCK = 1024

        grid = (N * G,)
        fused_gn_pool_kernel[grid](
            y, gamma, beta, scale, out,
            N, C, H, W, G, CPG,
            H_OUT, W_OUT,
            HW, GROUP_SIZE,
            POOL, self.clamp_min, self.clamp_max, self.eps,
            BLOCK,
            num_warps=8,
        )
        return out