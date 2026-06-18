import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_tanh_scale_bias_pool_kernel(
    x_ptr, w_ptr, cb_ptr, b_ptr, out_ptr,
    N, IC,
    IH: tl.constexpr, IW: tl.constexpr,
    OC: tl.constexpr,
    OH: tl.constexpr, OW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_PW: tl.constexpr,  # number of pooled cols per program
):
    # grid: (N, OC, PH * ceil(PW/BLOCK_PW))
    n = tl.program_id(0)
    oc = tl.program_id(1)
    ph_pw_tile = tl.program_id(2)

    pw_tiles = (PW + BLOCK_PW - 1) // BLOCK_PW
    ph = ph_pw_tile // pw_tiles
    pw_tile = ph_pw_tile % pw_tiles

    pw_offs = pw_tile * BLOCK_PW + tl.arange(0, BLOCK_PW)
    pw_mask = pw_offs < PW

    cb = tl.load(cb_ptr + oc)
    bb = tl.load(b_ptr + oc)

    max_val = tl.full((BLOCK_PW,), -1e30, dtype=tl.float32)

    # Loop over pool window
    for ki in tl.static_range(0, POOL):
        for kj in tl.static_range(0, POOL):
            oh = ph * POOL + ki
            ow = pw_offs * POOL + kj  # vector of length BLOCK_PW

            acc = tl.zeros((BLOCK_PW,), dtype=tl.float32)

            for ic in range(0, IC):
                for kh in tl.static_range(0, KH):
                    for kw in tl.static_range(0, KW):
                        ih = oh + kh
                        iw = ow + kw
                        in_offset = ((n * IC + ic) * IH + ih) * IW + iw
                        x_val = tl.load(x_ptr + in_offset, mask=pw_mask, other=0.0)
                        w_offset = ((oc * IC + ic) * KH + kh) * KW + kw
                        w_val = tl.load(w_ptr + w_offset)
                        acc += x_val * w_val

            acc = acc + cb
            # tanh via exp
            e2 = tl.exp(2.0 * acc)
            t = (e2 - 1.0) / (e2 + 1.0)
            v = t * SCALE + bb
            max_val = tl.maximum(max_val, v)

    out_offset = ((n * OC + oc) * PH + ph) * PW + pw_offs
    tl.store(out_ptr + out_offset, max_val, mask=pw_mask)


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

        if PH * POOL != OH or PW * POOL != OW:
            y = self.conv(x)
            y = torch.tanh(y) * self.scaling_factor + self.bias
            return self.max_pool(y)

        out = torch.empty((N, OC, PH, PW), device=x.device, dtype=x.dtype)
        bias_flat = self.bias.view(-1).contiguous()

        BLOCK_PW = 32
        pw_tiles = (PW + BLOCK_PW - 1) // BLOCK_PW
        grid = (N, OC, PH * pw_tiles)
        fused_conv_tanh_scale_bias_pool_kernel[grid](
            x, self.conv.weight, self.conv.bias, bias_flat, out,
            N, IC,
            IH, IW,
            OC,
            OH, OW,
            PH, PW,
            KH, KW,
            POOL,
            self.scaling_factor,
            BLOCK_PW,
            num_warps=4,
            num_stages=2,
        )
        return out