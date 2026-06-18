import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gn_stats_kernel(
    x_ptr,
    mean_ptr,
    rstd_ptr,
    N, C, H, W,
    G, CPG, GROUP_SIZE,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)
    group_start = pid_n * C * H * W + pid_g * CPG * H * W

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
    mean = s / GROUP_SIZE
    var = s2 / GROUP_SIZE - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + pid_n * G + pid_g, mean)
    tl.store(rstd_ptr + pid_n * G + pid_g, rstd)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 8}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 32, 'BLOCK_OW': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 64}, num_warps=4, num_stages=2),
    ],
    key=['H_out', 'W_out', 'P'],
)
@triton.jit
def fused_apply_kernel(
    x_ptr,
    mean_ptr,
    rstd_ptr,
    gamma_ptr,
    beta_ptr,
    scale_ptr,
    out_ptr,
    N, C, H, W,
    H_out, W_out,
    G, CPG,
    clamp_min,
    clamp_max,
    P: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_ow = tl.program_id(2)

    n = pid_nc // C
    c = pid_nc % C
    g = c // CPG

    mean = tl.load(mean_ptr + n * G + g)
    rstd = tl.load(rstd_ptr + n * G + g)
    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)
    s_scale = tl.load(scale_ptr + c)

    a_coef = rstd * gamma * s_scale
    b_coef = (beta - mean * rstd * gamma) * s_scale

    oh_offsets = pid_oh * BLOCK_OH + tl.arange(0, BLOCK_OH)
    ow_offsets = pid_ow * BLOCK_OW + tl.arange(0, BLOCK_OW)
    oh_mask = oh_offsets < H_out
    ow_mask = ow_offsets < W_out

    ih_start = oh_offsets[:, None] * P  # [BLOCK_OH, 1]
    iw_start = ow_offsets[None, :] * P  # [1, BLOCK_OW]

    ch_start = n * C * H * W + c * H * W

    max_val = tl.full([BLOCK_OH, BLOCK_OW], -float('inf'), dtype=tl.float32)
    out_mask = oh_mask[:, None] & ow_mask[None, :]

    for ph in tl.static_range(0, P):
        for pw in tl.static_range(0, P):
            ih = ih_start + ph
            iw = iw_start + pw
            v = tl.load(x_ptr + ch_start + ih * W + iw, mask=out_mask, other=-float('inf'))
            val = v * a_coef + b_coef
            max_val = tl.maximum(max_val, val)

    max_val = tl.minimum(tl.maximum(max_val, clamp_min), clamp_max)
    out_base = n * C * H_out * W_out + c * H_out * W_out
    out_offsets = oh_offsets[:, None] * W_out + ow_offsets[None, :]
    tl.store(out_ptr + out_base + out_offsets, max_val, mask=out_mask)


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
        mean = torch.empty((N, G), device=x.device, dtype=torch.float32)
        rstd = torch.empty((N, G), device=x.device, dtype=torch.float32)

        BLOCK_SIZE = 1024
        if GROUP_SIZE < 1024:
            BLOCK_SIZE = 256
        elif GROUP_SIZE >= 8192:
            BLOCK_SIZE = 2048

        scale_flat = self.scale.view(-1).contiguous()

        _gn_stats_kernel[(N, G)](
            x, mean, rstd,
            N, C, H, W,
            G, CPG, GROUP_SIZE,
            self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
        )

        grid = lambda META: (N * C, triton.cdiv(H_out, META['BLOCK_OH']), triton.cdiv(W_out, META['BLOCK_OW']))
        fused_apply_kernel[grid](
            x, mean, rstd,
            self.group_norm.weight, self.group_norm.bias, scale_flat, out,
            N, C, H, W,
            H_out, W_out,
            G, CPG,
            self.clamp_min, self.clamp_max,
            P=P,
        )
        return out