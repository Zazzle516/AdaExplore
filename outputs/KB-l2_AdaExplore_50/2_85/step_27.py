import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 256}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OH', 'OW', 'K_TOTAL'],
)
@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    K_TOTAL: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oh = sp_offs // OW
    ow = sp_offs % OW

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (OH * OW)

    # Build im2col tile [K_TOTAL, BLOCK_SP] and weight tile [BLOCK_OC, K_TOTAL]
    k_idx = tl.arange(0, K_TOTAL)
    ic_k = k_idx // (KH * KW)
    kh_k = (k_idx % (KH * KW)) // KW
    kw_k = k_idx % KW

    # x tile: gather [K_TOTAL, BLOCK_SP]
    ih = oh[None, :] + kh_k[:, None]  # [K_TOTAL, BLOCK_SP]
    iw = ow[None, :] + kw_k[:, None]
    x_off = (pid_n * IC * IH * IW
             + ic_k[:, None] * (IH * IW)
             + ih * IW + iw)
    x_mask = sp_mask[None, :]
    x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

    # w tile: [BLOCK_OC, K_TOTAL]
    w_off = oc_offs[:, None] * K_TOTAL + k_idx[None, :]
    w_mask = oc_mask[:, None]
    w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

    acc = tl.dot(w_tile, x_tile, out_dtype=tl.float32)

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += bias[:, None]

    out_off = pid_n * OC * OH * OW + oc_offs[:, None] * (OH * OW) + sp_offs[None, :]
    mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask)


@triton.jit
def gn_stats_kernel(
    x_ptr, mean_ptr, rstd_ptr,
    N, C, H, W,
    CPG: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)
    G = C // CPG

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
    inv_n = 1.0 / GROUP_SIZE
    mean = s * inv_n
    var = s2 * inv_n - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + pid_n * G + pid_g, mean)
    tl.store(rstd_ptr + pid_n * G + pid_g, rstd)


@triton.jit
def maxpool_epilogue_kernel(
    x_ptr, mean_ptr, rstd_ptr, gamma_ptr, beta_ptr, out_ptr,
    N, C, H, W,
    H_out, W_out,
    CPG: tl.constexpr,
    P: tl.constexpr,
    clamp_min,
    clamp_max,
    H_OUT_TILE: tl.constexpr,
    W_OUT_TILE: tl.constexpr,
):
    # one program per (n, c, tile_h, tile_w)
    pid_nc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    n = pid_nc // C
    c = pid_nc % C
    G = C // CPG
    g = c // CPG

    mean = tl.load(mean_ptr + n * G + g)
    rstd = tl.load(rstd_ptr + n * G + g)
    gm = tl.load(gamma_ptr + c)
    bt = tl.load(beta_ptr + c)
    a_coef = rstd * gm
    b_coef = bt - mean * rstd * gm

    ch_start = n * C * H * W + c * H * W

    oh_off = pid_h * H_OUT_TILE + tl.arange(0, H_OUT_TILE)
    ow_off = pid_w * W_OUT_TILE + tl.arange(0, W_OUT_TILE)
    oh_mask = oh_off < H_out
    ow_mask = ow_off < W_out

    max_val = tl.full([H_OUT_TILE, W_OUT_TILE], -float('inf'), dtype=tl.float32)
    for ph in tl.static_range(0, P):
        for pw in tl.static_range(0, P):
            ih = oh_off * P + ph
            iw = ow_off * P + pw
            ih_mask = ih < H
            iw_mask = iw < W
            offs = ih[:, None] * W + iw[None, :]
            m = (ih_mask[:, None] & iw_mask[None, :]) & (oh_mask[:, None] & ow_mask[None, :])
            v = tl.load(x_ptr + ch_start + offs, mask=m, other=-float('inf'))
            val = v * a_coef + b_coef
            max_val = tl.maximum(max_val, val)

    max_val = tl.minimum(tl.maximum(max_val, clamp_min), clamp_max)

    out_off = (n * C * H_out * W_out + c * H_out * W_out
               + oh_off[:, None] * W_out + ow_off[None, :])
    out_mask = oh_mask[:, None] & ow_mask[None, :]
    tl.store(out_ptr + out_off, max_val, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape,
                 maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
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
        x = x.contiguous().cuda()
        N = x.shape[0]
        IC = self.in_channels
        IH = x.shape[2]
        IW = x.shape[3]
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1

        conv_out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()

        K_TOTAL = IC * KH * KW

        grid_conv = lambda META: (N, triton.cdiv(OC, META['BLOCK_OC']), triton.cdiv(OH * OW, META['BLOCK_SP']))
        conv2d_kernel[grid_conv](
            x, w, b, conv_out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH=KH, KW=KW,
            K_TOTAL=K_TOTAL,
        )

        scale_flat = self.scale.view(-1).contiguous()
        gamma_eff = (self.group_norm.weight * scale_flat).contiguous()
        beta_eff = (self.group_norm.bias * scale_flat).contiguous()

        P = self.maxpool_kernel_size
        H_out = OH // P
        W_out = OW // P

        G = self.num_groups
        CPG = OC // G
        GROUP_SIZE = CPG * OH * OW

        mean_t = torch.empty((N, G), device=x.device, dtype=torch.float32)
        rstd_t = torch.empty((N, G), device=x.device, dtype=torch.float32)

        BLOCK_SIZE = 2048

        grid_stats = (N, G)
        gn_stats_kernel[grid_stats](
            conv_out, mean_t, rstd_t,
            N, OC, OH, OW,
            CPG, GROUP_SIZE,
            self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8, num_stages=2,
        )

        out = torch.empty((N, OC, H_out, W_out), device=x.device, dtype=x.dtype)

        H_OUT_TILE = 8
        W_OUT_TILE = 32

        grid_ep = (N * OC, triton.cdiv(H_out, H_OUT_TILE), triton.cdiv(W_out, W_OUT_TILE))
        maxpool_epilogue_kernel[grid_ep](
            conv_out, mean_t, rstd_t, gamma_eff, beta_eff, out,
            N, OC, OH, OW,
            H_out, W_out,
            CPG,
            P,
            self.clamp_min, self.clamp_max,
            H_OUT_TILE=H_OUT_TILE,
            W_OUT_TILE=W_OUT_TILE,
            num_warps=8, num_stages=2,
        )
        return out