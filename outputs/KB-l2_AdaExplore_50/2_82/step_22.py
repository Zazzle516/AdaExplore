import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_PH': 4, 'BLOCK_PW': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_PH': 4, 'BLOCK_PW': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PH': 4, 'BLOCK_PW': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PH': 2, 'BLOCK_PW': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_PH': 8, 'BLOCK_PW': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_PH': 4, 'BLOCK_PW': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PH': 4, 'BLOCK_PW': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_PH': 4, 'BLOCK_PW': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_PH': 2, 'BLOCK_PW': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PH': 2, 'BLOCK_PW': 16}, num_warps=8, num_stages=2),
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
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_p = tl.program_id(2)

    # Number of pool tiles along W
    num_pw_tiles = (PW + BLOCK_PW - 1) // BLOCK_PW
    pid_ph = pid_p // num_pw_tiles
    pid_pw = pid_p % num_pw_tiles

    # Tile sizes (with conv window expansion)
    HCONV: tl.constexpr = BLOCK_PH * POOL + 0  # OH tile rows = BLOCK_PH * POOL
    WCONV: tl.constexpr = BLOCK_PW * POOL + 0
    PP: tl.constexpr = POOL * POOL
    NUM_POOLS: tl.constexpr = BLOCK_PH * BLOCK_PW
    HW_TILE: tl.constexpr = HCONV * WCONV  # full conv tile size

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # output (conv) coordinate range for this tile
    oh_base = pid_ph * BLOCK_PH * POOL
    ow_base = pid_pw * BLOCK_PW * POOL

    h_local = tl.arange(0, HCONV)        # 0..HCONV-1
    w_local = tl.arange(0, WCONV)        # 0..WCONV-1
    oh = oh_base + h_local               # [HCONV]
    ow = ow_base + w_local               # [WCONV]
    oh_mask = oh < OH                    # [HCONV]
    ow_mask = ow < OW                    # [WCONV]

    # Load conv bias / extra bias
    cb = tl.load(cb_ptr + oc_offs, mask=oc_mask, other=0.0)
    bb = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    # accumulator [BLOCK_OC, HCONV*WCONV]
    acc = tl.zeros((BLOCK_OC, HCONV * WCONV), dtype=tl.float32)

    # flatten (h,w) of conv tile
    flat = tl.arange(0, HW_TILE)
    fh = flat // WCONV
    fw = flat % WCONV

    base_x = pid_n * IC * IH * IW

    for ic in range(0, IC):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = oh_base + fh + kh
                iw = ow_base + fw + kw
                in_off = base_x + ic * IH * IW + ih * IW + iw
                # We don't need bounds check on ih/iw because conv output is OH x OW = IH-KH+1, so ih<IH, iw<IW always for valid oh/ow
                # But the tile may extend beyond OH/OW; loads beyond still in-bounds for input as long as ih<IH, iw<IW.
                # Mask: only valid spatial positions within conv output bounds
                hmask = (oh_base + fh) < OH
                wmask = (ow_base + fw) < OW
                load_mask = hmask & wmask
                x_val = tl.load(x_ptr + in_off, mask=load_mask, other=0.0)  # [HW_TILE]

                w_off = oc_offs * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                acc += w_val[:, None] * x_val[None, :]

    # Apply conv bias, tanh, scale, bias
    acc = acc + cb[:, None]
    t = 2.0 * acc
    tanh_val = 2.0 / (1.0 + tl.exp(-t)) - 1.0
    val = tanh_val * scale + bb[:, None]  # [BLOCK_OC, HW_TILE]

    # mask invalid positions
    h_valid = (oh_base + fh) < OH
    w_valid = (ow_base + fw) < OW
    valid = h_valid & w_valid
    neg_inf = float('-inf')
    val = tl.where(valid[None, :], val, neg_inf)

    # Reshape to [BLOCK_OC, BLOCK_PH, POOL, BLOCK_PW, POOL] and reduce
    val = tl.reshape(val, (BLOCK_OC, BLOCK_PH, POOL, BLOCK_PW, POOL))
    val = tl.max(val, axis=4)   # [BLOCK_OC, BLOCK_PH, POOL, BLOCK_PW]
    val = tl.max(val, axis=2)   # [BLOCK_OC, BLOCK_PH, BLOCK_PW]

    # Store output
    ph_idx = pid_ph * BLOCK_PH + tl.arange(0, BLOCK_PH)
    pw_idx = pid_pw * BLOCK_PW + tl.arange(0, BLOCK_PW)
    ph_mask = ph_idx < PH
    pw_mask = pw_idx < PW

    out_off = (
        (pid_n * OC + oc_offs[:, None, None]) * (PH * PW)
        + ph_idx[None, :, None] * PW
        + pw_idx[None, None, :]
    )
    store_mask = oc_mask[:, None, None] & ph_mask[None, :, None] & pw_mask[None, None, :]
    tl.store(out_ptr + out_off, val, mask=store_mask)


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

        def grid(meta):
            num_pw_tiles = triton.cdiv(PW, meta['BLOCK_PW'])
            num_ph_tiles = triton.cdiv(PH, meta['BLOCK_PH'])
            return (
                N,
                triton.cdiv(OC, meta['BLOCK_OC']),
                num_ph_tiles * num_pw_tiles,
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