import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=2),
    ],
    key=['GROUP_SIZE_K', 'GS_C'],
)
@triton.jit
def gn_stats_kernel(
    x_ptr,
    mean_ptr,
    rstd_ptr,
    N, C,
    HW,
    CPG_C: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,
    GS_C: tl.constexpr,
    EPS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)
    G = tl.num_programs(1)

    group_start = pid_n * C * HW + pid_g * CPG_C * HW

    idx = tl.arange(0, GS_C)
    mask = idx < GROUP_SIZE_K
    v = tl.load(x_ptr + group_start + idx, mask=mask, other=0.0)
    s = tl.sum(v, axis=0)
    s2 = tl.sum(v * v, axis=0)

    inv_n = 1.0 / GROUP_SIZE_K
    mean = s * inv_n
    var = s2 * inv_n - mean * mean
    rstd = 1.0 / tl.sqrt(var + EPS)

    out_idx = pid_n * G + pid_g
    tl.store(mean_ptr + out_idx, mean)
    tl.store(rstd_ptr + out_idx, rstd)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=['W_C', 'P'],
)
@triton.jit
def fused_norm_scale_maxpool_clamp_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    rstd_ptr,
    out_ptr,
    C, H, W,
    H_out, W_out,
    HW,
    HOUT_WOUT,
    G,
    CPG: tl.constexpr,
    P: tl.constexpr,
    W_C: tl.constexpr,
    WOUT_C: tl.constexpr,
    OH_BLOCK: tl.constexpr,
    CLAMP_MIN: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_oh_blk = tl.program_id(2)

    g = pid_c // CPG
    stat_idx = pid_n * G + g
    mean = tl.load(mean_ptr + stat_idx)
    rstd = tl.load(rstd_ptr + stat_idx)

    g_w = tl.load(gamma_ptr + pid_c)
    b_w = tl.load(beta_ptr + pid_c)
    a_coef = rstd * g_w
    b_coef = b_w - mean * a_coef

    ch_in_start = pid_n * C * HW + pid_c * HW
    ch_out_start = pid_n * C * HOUT_WOUT + pid_c * HOUT_WOUT

    col_idx = tl.arange(0, W_C)
    col_mask = col_idx < W
    ow_idx = tl.arange(0, WOUT_C)
    ow_mask = ow_idx < W_out

    oh_start = pid_oh_blk * OH_BLOCK
    for oh_local in tl.static_range(0, OH_BLOCK):
        oh = oh_start + oh_local
        valid_oh = oh < H_out
        ih_base = oh * P

        # Load P rows in one go using 2D arange
        ph_idx = tl.arange(0, P)
        ih = ih_base + ph_idx[:, None]
        row_mask = (ih < H) & col_mask[None, :] & valid_oh
        addrs = ch_in_start + ih * W + col_idx[None, :]
        vv = tl.load(x_ptr + addrs, mask=row_mask, other=-float('inf'))
        val = vv * a_coef + b_coef  # [P, W_C]

        # max over P (axis=0) -> [W_C]
        vmax = tl.max(val, axis=0)

        # reshape [W_C] -> [WOUT_C, P], max over axis=1
        vmax_2d = tl.reshape(vmax, [WOUT_C, P])
        hmax = tl.max(vmax_2d, axis=1)

        hmax = tl.minimum(tl.maximum(hmax, CLAMP_MIN), CLAMP_MAX)

        store_mask = ow_mask & valid_oh
        tl.store(out_ptr + ch_out_start + oh * W_out + ow_idx, hmax, mask=store_mask)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.num_groups = num_groups
        self.out_channels = out_channels
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv(x)
        x = x.contiguous()
        N, C, H, W = x.shape
        P = self.maxpool_kernel_size
        H_out = H // P
        W_out = W // P

        G = self.num_groups
        CPG = C // G
        GROUP_SIZE = CPG * H * W

        out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

        scale_flat = self.scale.view(-1).contiguous()
        gamma_folded = (self.group_norm.weight * scale_flat).contiguous()
        beta_folded = (self.group_norm.bias * scale_flat).contiguous()

        W_C = _next_pow2(W)
        WOUT_C = W_C // P

        GS_C = _next_pow2(GROUP_SIZE)

        mean = torch.empty((N, G), device=x.device, dtype=torch.float32)
        rstd = torch.empty((N, G), device=x.device, dtype=torch.float32)

        gn_stats_kernel[(N, G)](
            x, mean, rstd,
            N, C,
            H * W,
            CPG_C=CPG,
            GROUP_SIZE_K=GROUP_SIZE,
            GS_C=GS_C,
            EPS=self.eps,
        )

        # Choose OH_BLOCK: split H_out into chunks for parallelism
        if H_out >= 8:
            OH_BLOCK = 4
        elif H_out >= 4:
            OH_BLOCK = 2
        else:
            OH_BLOCK = 1
        oh_blocks = (H_out + OH_BLOCK - 1) // OH_BLOCK

        fused_norm_scale_maxpool_clamp_kernel[(N, C, oh_blocks)](
            x, gamma_folded, beta_folded, mean, rstd, out,
            C, H, W,
            H_out, W_out,
            H * W,
            H_out * W_out,
            G,
            CPG=CPG,
            P=P,
            W_C=W_C,
            WOUT_C=WOUT_C,
            OH_BLOCK=OH_BLOCK,
            CLAMP_MIN=self.clamp_min,
            CLAMP_MAX=self.clamp_max,
        )
        return out