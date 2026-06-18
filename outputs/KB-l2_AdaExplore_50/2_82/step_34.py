import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_PW': 8, 'BLOCK_OC': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_PW': 8, 'BLOCK_OC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_PW': 8, 'BLOCK_OC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_PW': 16, 'BLOCK_OC': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_PW': 16, 'BLOCK_OC': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_PW': 4, 'BLOCK_OC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_PW': 4, 'BLOCK_OC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_PW': 8, 'BLOCK_OC': 32}, num_warps=8, num_stages=3),
    ],
    key=['IC', 'OC', 'IH', 'IW', 'KH', 'KW', 'POOL'],
)
@triton.jit
def fused_conv_tanh_scale_bias_pool_kernel(
    x_ptr, w_ptr, cb_ptr, b_ptr, out_ptr,
    N, IC,
    IH: tl.constexpr, IW: tl.constexpr,
    OC: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_PW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    n = tl.program_id(0)
    oc_tile = tl.program_id(1)
    phpw = tl.program_id(2)

    pw_tiles = (PW + BLOCK_PW - 1) // BLOCK_PW
    ph = phpw // pw_tiles
    pw_t = phpw % pw_tiles

    oc_offs = oc_tile * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    NUM_OUT: tl.constexpr = BLOCK_PW * POOL * POOL
    p = tl.arange(0, NUM_OUT)
    pw_idx = p // (POOL * POOL)
    inner2 = p % (POOL * POOL)
    r = inner2 // POOL
    kj = inner2 % POOL
    pw_global = pw_t * BLOCK_PW + pw_idx
    p_mask = pw_global < PW

    oh = ph * POOL + r
    ow = pw_global * POOL + kj

    acc = tl.zeros((BLOCK_OC, NUM_OUT), dtype=tl.float32)

    for ic in range(0, IC):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = oh + kh
                iw = ow + kw
                x_off = ((n * IC + ic) * IH + ih) * IW + iw
                x_val = tl.load(x_ptr + x_off, mask=p_mask, other=0.0)
                w_off = ((oc_offs * IC + ic) * KH + kh) * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                acc += w_val[:, None] * x_val[None, :]

    cb = tl.load(cb_ptr + oc_offs, mask=oc_mask, other=0.0)
    bb = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + cb[:, None]
    e2 = tl.exp(2.0 * acc)
    t = (e2 - 1.0) / (e2 + 1.0)
    v = t * SCALE + bb[:, None]

    v_r = tl.reshape(v, (BLOCK_OC, BLOCK_PW, POOL * POOL))
    out_v = tl.max(v_r, axis=2)

    pw_out = pw_t * BLOCK_PW + tl.arange(0, BLOCK_PW)
    pw_out_mask = pw_out < PW
    out_off = ((n * OC + oc_offs[:, None]) * PH + ph) * PW + pw_out[None, :]
    out_mask = oc_mask[:, None] & pw_out_mask[None, :]
    tl.store(out_ptr + out_off, out_v, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = float(scaling_factor)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1
        POOL = self.pool_kernel_size
        PH = OH // POOL
        PW = OW // POOL

        out = torch.empty((N, OC, PH, PW), device=x.device, dtype=x.dtype)
        bias_flat = self.bias.view(-1).contiguous()

        def grid(meta):
            BLOCK_PW = meta['BLOCK_PW']
            BLOCK_OC = meta['BLOCK_OC']
            pw_tiles = (PW + BLOCK_PW - 1) // BLOCK_PW
            oc_tiles = (OC + BLOCK_OC - 1) // BLOCK_OC
            return (N, oc_tiles, PH * pw_tiles)

        fused_conv_tanh_scale_bias_pool_kernel[grid](
            x, self.conv.weight, self.conv.bias, bias_flat, out,
            N, IC,
            IH, IW,
            OC,
            PH, PW,
            KH, KW,
            POOL,
            self.scaling_factor,
        )
        return out