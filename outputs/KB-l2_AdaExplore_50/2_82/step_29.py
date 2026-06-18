import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PH': 4, 'BLOCK_PW': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PH': 4, 'BLOCK_PW': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PH': 8, 'BLOCK_PW': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PH': 8, 'BLOCK_PW': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PH': 8, 'BLOCK_PW': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PH': 8, 'BLOCK_PW': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PH': 16, 'BLOCK_PW': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_PH': 16, 'BLOCK_PW': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_PH': 8, 'BLOCK_PW': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_PH': 8, 'BLOCK_PW': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_PH': 4, 'BLOCK_PW': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PH': 4, 'BLOCK_PW': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PH': 4, 'BLOCK_PW': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_PH': 4, 'BLOCK_PW': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_PH': 4, 'BLOCK_PW': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_PH': 8, 'BLOCK_PW': 8}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'PH', 'PW', 'IC', 'KH', 'KW', 'POOL'],
)
@triton.jit
def fused_conv_tanh_scale_bias_pool_kernel(
    x_ptr, w_ptr, cb_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    PH, PW,
    scale,
    KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_PH: tl.constexpr,
    BLOCK_PW: tl.constexpr,
):
    # 2D pool tiling: each program owns (n, oc_tile, ph_tile*pw_tile region)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_p = tl.program_id(2)

    num_pw_tiles = (PW + BLOCK_PW - 1) // BLOCK_PW
    pid_ph = pid_p // num_pw_tiles
    pid_pw = pid_p % num_pw_tiles

    # output-conv tile size
    OH_TILE: tl.constexpr = BLOCK_PH * POOL
    OW_TILE: tl.constexpr = BLOCK_PW * POOL
    CONV_TILE: tl.constexpr = OH_TILE * OW_TILE

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # conv output coords within tile
    oh_local = tl.arange(0, OH_TILE)
    ow_local = tl.arange(0, OW_TILE)
    oh_global = pid_ph * OH_TILE + oh_local  # [OH_TILE]
    ow_global = pid_pw * OW_TILE + ow_local  # [OW_TILE]
    oh_mask = oh_global < OH
    ow_mask = ow_global < OW

    # accumulator [BLOCK_OC, OH_TILE, OW_TILE] flattened as [BLOCK_OC, CONV_TILE]
    acc = tl.zeros((BLOCK_OC, CONV_TILE), dtype=tl.float32)

    # combined oh,ow flat positions
    flat = tl.arange(0, CONV_TILE)
    oh_flat = flat // OW_TILE  # [CONV_TILE]
    ow_flat = flat % OW_TILE
    oh_g_flat = pid_ph * OH_TILE + oh_flat
    ow_g_flat = pid_pw * OW_TILE + ow_flat
    valid_flat = (oh_g_flat < OH) & (ow_g_flat < OW)

    base_x = pid_n * IC * IH * IW

    for ic in range(0, IC):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = oh_g_flat + kh
                iw = ow_g_flat + kw
                in_off = base_x + ic * IH * IW + ih * IW + iw
                x_val = tl.load(x_ptr + in_off, mask=valid_flat, other=0.0)  # [CONV_TILE]

                w_off = oc_offs * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                acc += w_val[:, None] * x_val[None, :]

    cb = tl.load(cb_ptr + oc_offs, mask=oc_mask, other=0.0)
    bb = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    acc = acc + cb[:, None]
    t = 2.0 * acc
    tanh_val = 2.0 / (1.0 + tl.exp(-t)) - 1.0
    val = tanh_val * scale + bb[:, None]

    neg_inf = float('-inf')
    val = tl.where(valid_flat[None, :], val, neg_inf)

    # reshape to [BLOCK_OC, BLOCK_PH, POOL, BLOCK_PW, POOL] then max over POOLs
    val = tl.reshape(val, (BLOCK_OC, BLOCK_PH, POOL, BLOCK_PW, POOL))
    val = tl.max(val, axis=4)  # [BLOCK_OC, BLOCK_PH, POOL, BLOCK_PW]
    val = tl.max(val, axis=2)  # [BLOCK_OC, BLOCK_PH, BLOCK_PW]

    # store
    ph_local = tl.arange(0, BLOCK_PH)
    pw_local = tl.arange(0, BLOCK_PW)
    ph_g = pid_ph * BLOCK_PH + ph_local
    pw_g = pid_pw * BLOCK_PW + pw_local
    ph_mask = ph_g < PH
    pw_mask = pw_g < PW

    # out shape: [N, OC, PH, PW]
    # offs: oc x ph x pw
    out_base = (pid_n * OC + oc_offs[:, None, None]) * (PH * PW)
    out_off = out_base + ph_g[None, :, None] * PW + pw_g[None, None, :]
    mask = oc_mask[:, None, None] & ph_mask[None, :, None] & pw_mask[None, None, :]
    tl.store(out_ptr + out_off, val, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scaling_factor = float(scaling_factor)
        self.pool_kernel_size = pool_kernel_size

        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.weight = nn.Parameter(conv.weight.detach().clone())
        self.conv_bias = nn.Parameter(conv.bias.detach().clone())
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, IH, IW = x.shape
        KH = KW = self.kernel_size
        OC = self.out_channels
        OH = IH - KH + 1
        OW = IW - KW + 1
        POOL = self.pool_kernel_size
        PH = OH // POOL
        PW = OW // POOL

        out = torch.empty((N, OC, PH, PW), device=x.device, dtype=x.dtype)

        bias_flat = self.bias.view(-1).contiguous()
        weight = self.weight.contiguous()
        conv_bias = self.conv_bias.contiguous()

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_OC']),
            triton.cdiv(PH, meta['BLOCK_PH']) * triton.cdiv(PW, meta['BLOCK_PW']),
        )

        fused_conv_tanh_scale_bias_pool_kernel[grid](
            x, weight, conv_bias, bias_flat, out,
            N, IC, IH, IW,
            OC, OH, OW,
            PH, PW,
            self.scaling_factor,
            KH, KW,
            POOL,
        )

        return out