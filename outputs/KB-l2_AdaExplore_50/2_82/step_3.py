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
    PH, PW,
    scale,
    KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # program ids: (n, oc_block, pool_block)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_p = tl.program_id(2)

    num_pools = PH * PW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)

    oc_mask = oc_offs < OC
    p_mask = p_offs < num_pools

    # decode pool position
    ph = p_offs // PW
    pw = p_offs % PW

    # load conv bias + bias for each oc
    cb = tl.load(cb_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    bb = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)   # [BLOCK_OC]

    # accumulator for max pool: [BLOCK_OC, BLOCK_P]
    neg_inf = float('-inf')
    max_acc = tl.full((BLOCK_OC, BLOCK_P), neg_inf, dtype=tl.float32)

    # loop over pool window
    for i in tl.static_range(0, POOL):
        for j in tl.static_range(0, POOL):
            # output coords in conv result
            oh = ph * POOL + i  # [BLOCK_P]
            ow = pw * POOL + j  # [BLOCK_P]

            # compute conv at (oh, ow) for each oc in block
            # acc shape: [BLOCK_OC, BLOCK_P]
            acc = tl.zeros((BLOCK_OC, BLOCK_P), dtype=tl.float32)

            for ic in range(0, IC):
                for kh in tl.static_range(0, KH):
                    for kw in tl.static_range(0, KW):
                        ih = oh + kh  # [BLOCK_P]
                        iw = ow + kw  # [BLOCK_P]
                        # input offset: n, ic, ih, iw
                        in_off = (pid_n * IC + ic) * IH * IW + ih * IW + iw
                        x_val = tl.load(x_ptr + in_off, mask=p_mask, other=0.0)  # [BLOCK_P]

                        # weight: [OC, IC, KH, KW]
                        w_off = oc_offs * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                        w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                        acc += w_val[:, None] * x_val[None, :]

            # add conv bias, tanh, scale, add bias
            acc = acc + cb[:, None]
            # tanh via sigmoid
            t = 2.0 * acc
            # tanh(x) = 2*sigmoid(2x) - 1
            tanh_val = 2.0 / (1.0 + tl.exp(-t)) - 1.0
            val = tanh_val * scale + bb[:, None]

            max_acc = tl.maximum(max_acc, val)

    # store output: [N, OC, PH, PW]
    out_off = (pid_n * OC + oc_offs[:, None]) * (PH * PW) + p_offs[None, :]
    mask = oc_mask[:, None] & p_mask[None, :]
    tl.store(out_ptr + out_off, max_acc, mask=mask)


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

        BLOCK_OC = 16
        BLOCK_P = 64

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(PH * PW, BLOCK_P))

        fused_conv_tanh_scale_bias_pool_kernel[grid](
            x, weight, conv_bias, bias_flat, out,
            N, IC, IH, IW,
            OC, OH, OW,
            PH, PW,
            self.scaling_factor,
            KH, KW,
            POOL,
            BLOCK_OC, BLOCK_P,
            num_warps=4,
            num_stages=2,
        )

        return out