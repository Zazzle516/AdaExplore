import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_gn_scale_maxpool_clamp_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    N, C, H, W,
    H_out, W_out,
    G, CPG,
    GROUP_SIZE,
    eps,
    clamp_min, clamp_max,
    P: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CPG_C: tl.constexpr,
    HW: tl.constexpr,
    HOUT_WOUT: tl.constexpr,
    W_C: tl.constexpr,
    WOUT_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_start = pid_n * C * HW + pid_g * CPG_C * HW

    sum_x = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    sum_x2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for offset in range(0, GROUP_SIZE, BLOCK_SIZE):
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < GROUP_SIZE
        v = tl.load(x_ptr + group_start + idx, mask=mask, other=0.0)
        sum_x += tl.where(mask, v, 0.0)
        sum_x2 += tl.where(mask, v * v, 0.0)

    s = tl.sum(sum_x, axis=0)
    s2 = tl.sum(sum_x2, axis=0)

    inv_n = 1.0 / GROUP_SIZE
    mean = s * inv_n
    var = s2 * inv_n - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    col_idx = tl.arange(0, W_C)
    col_mask = col_idx < W
    ow_idx = tl.arange(0, WOUT_C)
    ow_mask = ow_idx < W_out
    neg_inf = -float('inf')

    for c_local in range(0, CPG_C):
        c = pid_g * CPG_C + c_local
        g_w = tl.load(gamma_ptr + c)
        b_w = tl.load(beta_ptr + c)
        a_coef = rstd * g_w
        b_coef = b_w - mean * rstd * g_w

        ch_in_start = pid_n * C * HW + c * HW
        ch_out_start = pid_n * C * HOUT_WOUT + c * HOUT_WOUT

        for oh in range(0, H_out):
            ih_base = oh * P

            vmax = tl.full([W_C], neg_inf, dtype=tl.float32)
            for ph in range(0, P):
                ih = ih_base + ph
                row_mask = (ih < H) & col_mask
                v = tl.load(x_ptr + ch_in_start + ih * W + col_idx, mask=row_mask, other=neg_inf)
                val = v * a_coef + b_coef
                vmax = tl.maximum(vmax, val)

            vmax_2d = tl.reshape(vmax, [WOUT_C, P])
            hmax = tl.max(vmax_2d, axis=1)

            hmax = tl.minimum(tl.maximum(hmax, clamp_min), clamp_max)

            tl.store(out_ptr + ch_out_start + oh * W_out + ow_idx, hmax, mask=ow_mask)


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

        BLOCK_SIZE = 2048
        if GROUP_SIZE < 2048:
            BLOCK_SIZE = 256

        W_C = _next_pow2(W)
        WOUT_C = _next_pow2(W_out)
        if W_C != WOUT_C * P:
            WOUT_C = W_C // P

        grid = (N, G)
        fused_gn_scale_maxpool_clamp_kernel[grid](
            x, gamma_folded, beta_folded, out,
            N, C, H, W,
            H_out, W_out,
            G, CPG,
            GROUP_SIZE,
            self.eps,
            self.clamp_min, self.clamp_max,
            P=P,
            BLOCK_SIZE=BLOCK_SIZE,
            CPG_C=CPG,
            HW=H * W,
            HOUT_WOUT=H_out * W_out,
            W_C=W_C,
            WOUT_C=WOUT_C,
            num_warps=4,
            num_stages=2,
        )
        return out