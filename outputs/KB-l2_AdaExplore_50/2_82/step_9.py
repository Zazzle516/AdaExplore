import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_tanh_scale_bias_maxpool_kernel(
    x_ptr, w_ptr, cb_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    PH, PW,
    KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # program ids: (n, oc_block, p_block) where p indexes pooled output spatial
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_p = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    P_total = PH * PW
    p_mask = p_offs < P_total

    p_h = p_offs // PW  # pooled output h
    p_w = p_offs % PW   # pooled output w

    # The output of conv is OH x OW, pooled with non-overlapping POOL window
    # max over (POOL*POOL) conv-output positions per pooled-output element

    neg_inf = float('-inf')
    max_acc = tl.full((BLOCK_OC, BLOCK_P), neg_inf, dtype=tl.float32)

    # Load bias for these oc
    bias_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]

    for ph_i in tl.static_range(POOL):
        for pw_i in tl.static_range(POOL):
            # conv output position
            oh = p_h * POOL + ph_i  # [BLOCK_P]
            ow = p_w * POOL + pw_i  # [BLOCK_P]
            valid_oh = oh < OH
            valid_ow = ow < OW
            valid = valid_oh & valid_ow & p_mask

            # Compute conv at (n=pid_n, oc=oc_offs, oh, ow)
            # acc shape [BLOCK_OC, BLOCK_P]
            acc = tl.zeros((BLOCK_OC, BLOCK_P), dtype=tl.float32)

            for ic in range(IC):
                for kh in tl.static_range(KH):
                    for kw in tl.static_range(KW):
                        ih = oh + kh  # [BLOCK_P]
                        iw = ow + kw  # [BLOCK_P]
                        # input ptr: x[n, ic, ih, iw]
                        x_off = pid_n * (IC * IH * IW) + ic * (IH * IW) + ih[None, :] * IW + iw[None, :]
                        x_val = tl.load(x_ptr + x_off, mask=valid[None, :], other=0.0)  # [1, BLOCK_P]
                        # weight: w[oc, ic, kh, kw]
                        w_off = oc_offs * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                        w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                        acc += w_val[:, None] * x_val  # [BLOCK_OC, BLOCK_P]

            # Add conv bias
            cb = tl.load(cb_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
            acc = acc + cb[:, None]
            # tanh
            # tanh(x) = (exp(2x)-1)/(exp(2x)+1)
            e2x = tl.exp(2.0 * acc)
            tanh_val = (e2x - 1.0) / (e2x + 1.0)
            scaled = tanh_val * SCALE + bias_vals[:, None]

            # mask invalid positions to -inf
            scaled = tl.where(valid[None, :], scaled, neg_inf)
            max_acc = tl.maximum(max_acc, scaled)

    # store output: out[n, oc, p_h, p_w]
    out_off = pid_n * (OC * P_total) + oc_offs[:, None] * P_total + p_offs[None, :]
    out_mask = oc_mask[:, None] & p_mask[None, :]
    tl.store(out_ptr + out_off, max_acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = float(scaling_factor)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.pool_kernel_size = pool_kernel_size
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

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

        w = self.conv.weight.contiguous().cuda()
        cb = self.conv.bias.contiguous().cuda()
        bias_flat = self.bias.view(-1).contiguous().cuda()

        out = torch.empty((N, OC, PH, PW), device=x.device, dtype=torch.float32)

        BLOCK_OC = 16
        BLOCK_P = 64

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(PH * PW, BLOCK_P))

        fused_conv_tanh_scale_bias_maxpool_kernel[grid](
            x, w, cb, bias_flat, out,
            N, IC, IH, IW,
            OC, OH, OW,
            PH, PW,
            KH, KW,
            POOL,
            self.scaling_factor,
            BLOCK_OC, BLOCK_P,
            num_warps=4, num_stages=2,
        )
        return out