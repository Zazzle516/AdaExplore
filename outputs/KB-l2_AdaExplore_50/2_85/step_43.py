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
    CPG_C: tl.constexpr,
    HW: tl.constexpr,
    HOUT_WOUT: tl.constexpr,
    W_C: tl.constexpr,
    WOUT_C: tl.constexpr,
    GS_C: tl.constexpr,
    ROW_TILE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_start = pid_n * C * HW + pid_g * CPG_C * HW

    # Single-shot reduction since GS_C is the next power-of-two of GROUP_SIZE
    idx = tl.arange(0, GS_C)
    mask = idx < GROUP_SIZE
    v = tl.load(x_ptr + group_start + idx, mask=mask, other=0.0)
    s = tl.sum(v, axis=0)
    s2 = tl.sum(v * v, axis=0)

    inv_n = 1.0 / GROUP_SIZE
    mean = s * inv_n
    var = s2 * inv_n - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    neg_inf = -float('inf')
    col_idx = tl.arange(0, W_C)
    col_mask = col_idx < W
    ow_idx = tl.arange(0, WOUT_C)
    ow_mask = ow_idx < W_out

    for c_local in tl.static_range(0, CPG_C):
        c = pid_g * CPG_C + c_local
        g_w = tl.load(gamma_ptr + c)
        b_w = tl.load(beta_ptr + c)
        a_coef = rstd * g_w
        b_coef = b_w - mean * rstd * g_w

        ch_in_start = pid_n * C * HW + c * HW
        ch_out_start = pid_n * C * HOUT_WOUT + c * HOUT_WOUT

        # Process ROW_TILE output rows at a time.
        for oh_base in range(0, H_out, ROW_TILE):
            # For each output row in tile, do P input rows, vertical max.
            # We use a 2D tile [ROW_TILE, W_C] for the vertical max accumulator.
            row_off = tl.arange(0, ROW_TILE)
            vmax = tl.full([ROW_TILE, W_C], neg_inf, dtype=tl.float32)
            oh_vec = oh_base + row_off  # [ROW_TILE]
            oh_valid = oh_vec < H_out  # [ROW_TILE]

            for ph in tl.static_range(0, P):
                ih = oh_vec[:, None] * P + ph  # [ROW_TILE, 1]
                in_mask = oh_valid[:, None] & col_mask[None, :] & (ih < H)
                offs = ch_in_start + ih * W + col_idx[None, :]
                vv = tl.load(x_ptr + offs, mask=in_mask, other=neg_inf)
                val = vv * a_coef + b_coef
                vmax = tl.maximum(vmax, val)

            # Horizontal pool: reshape [ROW_TILE, WOUT_C, P] -> max axis 2
            vmax_3d = tl.reshape(vmax, [ROW_TILE, WOUT_C, P])
            hmax = tl.max(vmax_3d, axis=2)  # [ROW_TILE, WOUT_C]
            hmax = tl.minimum(tl.maximum(hmax, clamp_min), clamp_max)

            out_offs = ch_out_start + oh_vec[:, None] * W_out + ow_idx[None, :]
            store_mask = oh_valid[:, None] & ow_mask[None, :]
            tl.store(out_ptr + out_offs, hmax, mask=store_mask)


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
        WOUT_C = _next_pow2(W_out)
        if W_C != WOUT_C * P:
            WOUT_C = W_C // P

        GS_C = _next_pow2(GROUP_SIZE)

        # pick num_warps
        if GS_C >= 16384:
            num_warps = 16
        elif GS_C >= 4096:
            num_warps = 8
        else:
            num_warps = 4

        # Choose ROW_TILE
        ROW_TILE = 4
        if H_out < ROW_TILE:
            # find largest power of 2 <= H_out
            r = 1
            while r * 2 <= H_out:
                r *= 2
            ROW_TILE = max(r, 1)

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
            CPG_C=CPG,
            HW=H * W,
            HOUT_WOUT=H_out * W_out,
            W_C=W_C,
            WOUT_C=WOUT_C,
            GS_C=GS_C,
            ROW_TILE=ROW_TILE,
            num_warps=num_warps,
        )
        return out