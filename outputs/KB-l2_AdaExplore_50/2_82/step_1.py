import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_tanh_scale_bias_pool_kernel(
    x_ptr, w_ptr, cb_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    PH, PW,  # pooled output H, W
    KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_P: tl.constexpr,  # number of pooled output positions per program
):
    pid = tl.program_id(0)
    # grid: (N * OC, ceil(PH*PW / BLOCK_P))
    noc = tl.program_id(0)
    tile = tl.program_id(1)

    n = noc // OC
    oc = noc % OC

    p_offs = tile * BLOCK_P + tl.arange(0, BLOCK_P)
    p_mask = p_offs < (PH * PW)

    ph = p_offs // PW
    pw = p_offs % PW

    # bias for this oc
    cb = tl.load(cb_ptr + oc)  # conv bias
    bb = tl.load(b_ptr + oc)   # extra bias

    # accumulator for max
    neg_large = tl.full((BLOCK_P,), -1e30, dtype=tl.float32)
    max_val = neg_large

    # iterate over the POOL x POOL pooling window
    for ki in tl.static_range(0, POOL):
        for kj in tl.static_range(0, POOL):
            oh = ph * POOL + ki  # output spatial of conv
            ow = pw * POOL + kj

            # compute conv at (n, oc, oh, ow)
            acc = tl.zeros((BLOCK_P,), dtype=tl.float32)
            for ic in range(0, IC):
                for kh in tl.static_range(0, KH):
                    for kw in tl.static_range(0, KW):
                        ih = oh + kh
                        iw = ow + kw
                        # load input
                        in_offset = ((n * IC + ic) * IH + ih) * IW + iw
                        x_val = tl.load(x_ptr + in_offset, mask=p_mask, other=0.0)
                        w_offset = ((oc * IC + ic) * KH + kh) * KW + kw
                        w_val = tl.load(w_ptr + w_offset)
                        acc += x_val * w_val
            acc = acc + cb
            # tanh
            t = (tl.exp(acc) - tl.exp(-acc)) / (tl.exp(acc) + tl.exp(-acc))
            v = t * SCALE + bb
            max_val = tl.maximum(max_val, v)

    out_offset = ((n * OC + oc) * PH + ph) * PW + pw
    tl.store(out_ptr + out_offset, max_val, mask=p_mask)


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

        # If shapes don't divide evenly, fallback
        if PH * POOL != OH or PW * POOL != OW:
            y = self.conv(x)
            y = torch.tanh(y) * self.scaling_factor + self.bias
            return self.max_pool(y)

        out = torch.empty((N, OC, PH, PW), device=x.device, dtype=x.dtype)
        bias_flat = self.bias.view(-1).contiguous()

        BLOCK_P = 64
        grid = (N * OC, triton.cdiv(PH * PW, BLOCK_P))
        fused_conv_tanh_scale_bias_pool_kernel[grid](
            x, self.conv.weight, self.conv.bias, bias_flat, out,
            N, IC, IH, IW,
            OC, OH, OW,
            PH, PW,
            KH, KW,
            POOL,
            self.scaling_factor,
            BLOCK_P,
            num_warps=4,
        )
        return out