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
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=3),
    ],
    key=['C_IN', 'C_OUT', 'KH', 'KW', 'H_OUT', 'W_OUT'],
)
@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, H_IN, W_IN,
    H_OUT, W_OUT,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
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

    x_base = pid_n * (C_IN * H_IN * W_IN)

    for ic in tl.static_range(0, C_IN):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = oh + kh
                iw = ow + kw
                w_idx = oc_offs * (C_IN * KH * KW) + ic * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)
                x_idx = x_base + ic * (H_IN * W_IN) + ih * W_IN + iw
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
        N, H_IN, W_IN,
        H_OUT, W_OUT,
        C_IN, C_OUT, KH, KW,
    )
    return out


@triton.jit
def gn_stats_kernel(
    x_ptr, mean_ptr, rstd_ptr,
    GROUP_SIZE: tl.constexpr,
    G: tl.constexpr,
    CHW: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G
    base = n * CHW + g * GROUP_SIZE

    sum_ = tl.zeros((BLOCK,), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK,), dtype=tl.float32)

    for off in range(0, GROUP_SIZE, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < GROUP_SIZE
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        sum_ += v
        sum_sq += v * v

    s = tl.sum(sum_, axis=0)
    ssq = tl.sum(sum_sq, axis=0)
    mean = s / GROUP_SIZE
    var = ssq / GROUP_SIZE - mean * mean
    rstd = 1.0 / tl.sqrt(var + EPS)

    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)


@triton.jit
def gn_scale_maxpool_clamp_kernel(
    x_ptr, gamma_ptr, beta_ptr, scale_ptr, mean_ptr, rstd_ptr, out_ptr,
    N, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    G: tl.constexpr, CPG: tl.constexpr,
    H_OUT: tl.constexpr, W_OUT: tl.constexpr,
    POOL: tl.constexpr,
    CLAMP_MIN: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)

    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    oh = hw_offs // W_OUT
    ow = hw_offs % W_OUT
    hw_mask = hw_offs < (H_OUT * W_OUT)

    g = pid_c // CPG
    gn_idx = pid_n * G + g
    mean = tl.load(mean_ptr + gn_idx)
    rstd = tl.load(rstd_ptr + gn_idx)
    gamma = tl.load(gamma_ptr + pid_c)
    beta = tl.load(beta_ptr + pid_c)
    scale = tl.load(scale_ptr + pid_c)

    a = rstd * gamma * scale
    b = beta * scale - mean * rstd * gamma * scale

    acc = tl.full((BLOCK_HW,), -float('inf'), dtype=tl.float32)

    base = pid_n * (C * H * W) + pid_c * (H * W)

    for ph in tl.static_range(0, POOL):
        for pw in tl.static_range(0, POOL):
            ih = oh * POOL + ph
            iw = ow * POOL + pw
            x_idx = base + ih * W + iw
            v = tl.load(x_ptr + x_idx, mask=hw_mask, other=-float('inf'))
            acc = tl.maximum(acc, v)

    # If a < 0, the affine reverses ordering. To keep correctness for any sign of a,
    # apply affine and recompute via separate path. But gamma*scale*rstd sign matters.
    # Since maxpool then affine vs affine then maxpool differ when a<0.
    # Use the safe approach: compute affine before max in inner loop if a could be negative.
    # We'll handle this by branching at compile via tl.where on sign.
    acc = acc * a + b
    # If a < 0, we need actually min over window then affine. So redo properly:
    # We'll do the safe path below regardless when a < 0.
    acc = tl.minimum(tl.maximum(acc, CLAMP_MIN), CLAMP_MAX)

    out_idx = pid_n * (C * H_OUT * W_OUT) + pid_c * (H_OUT * W_OUT) + hw_offs
    tl.store(out_ptr + out_idx, acc, mask=hw_mask)


@triton.jit
def gn_scale_maxpool_clamp_safe_kernel(
    x_ptr, gamma_ptr, beta_ptr, scale_ptr, mean_ptr, rstd_ptr, out_ptr,
    N, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    G: tl.constexpr, CPG: tl.constexpr,
    H_OUT: tl.constexpr, W_OUT: tl.constexpr,
    POOL: tl.constexpr,
    CLAMP_MIN: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)

    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    oh = hw_offs // W_OUT
    ow = hw_offs % W_OUT
    hw_mask = hw_offs < (H_OUT * W_OUT)

    g = pid_c // CPG
    gn_idx = pid_n * G + g
    mean = tl.load(mean_ptr + gn_idx)
    rstd = tl.load(rstd_ptr + gn_idx)
    gamma = tl.load(gamma_ptr + pid_c)
    beta = tl.load(beta_ptr + pid_c)
    scale = tl.load(scale_ptr + pid_c)

    a = rstd * gamma * scale
    b = beta * scale - mean * rstd * gamma * scale

    acc = tl.full((BLOCK_HW,), -float('inf'), dtype=tl.float32)

    base = pid_n * (C * H * W) + pid_c * (H * W)

    for ph in tl.static_range(0, POOL):
        for pw in tl.static_range(0, POOL):
            ih = oh * POOL + ph
            iw = ow * POOL + pw
            x_idx = base + ih * W + iw
            v = tl.load(x_ptr + x_idx, mask=hw_mask, other=-float('inf'))
            v = v * a + b
            acc = tl.maximum(acc, v)

    acc = tl.minimum(tl.maximum(acc, CLAMP_MIN), CLAMP_MAX)

    out_idx = pid_n * (C * H_OUT * W_OUT) + pid_c * (H_OUT * W_OUT) + hw_offs
    tl.store(out_ptr + out_idx, acc, mask=hw_mask)


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
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.eps = 1e-5

    def forward(self, x):
        x = x.cuda().contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()

        y = triton_conv2d(x, w, b)
        N, C, H, W = y.shape

        G = self.num_groups
        CPG = C // G
        mean = torch.empty(N * G, device=y.device, dtype=torch.float32)
        rstd = torch.empty(N * G, device=y.device, dtype=torch.float32)
        group_size = CPG * H * W
        BLOCK = 2048
        gn_stats_kernel[(N * G,)](
            y, mean, rstd,
            group_size, G, C * H * W,
            self.eps, BLOCK,
            num_warps=8,
        )

        POOL = self.maxpool_kernel_size
        H_OUT = H // POOL
        W_OUT = W // POOL
        out = torch.empty((N, C, H_OUT, W_OUT), device=y.device, dtype=y.dtype)

        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()
        scale = self.scale.view(-1).contiguous()

        BLOCK_HW = 1024
        grid = (N, C, triton.cdiv(H_OUT * W_OUT, BLOCK_HW))
        # Use safe variant (affine before max) to be correct for any sign of a.
        gn_scale_maxpool_clamp_safe_kernel[grid](
            y, gamma, beta, scale, mean, rstd, out,
            N, C, H, W, G, CPG,
            H_OUT, W_OUT,
            POOL, self.clamp_min, self.clamp_max,
            BLOCK_HW,
            num_warps=8,
        )
        return out