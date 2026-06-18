import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_pool_sigmoid_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, KH, KW,
    OH, OW,
    PH, PW,
    stride_xn, stride_xh, stride_xw, stride_xc,  # NHWC strides
    POOL: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    KH_C: tl.constexpr,
    KW_C: tl.constexpr,
    IC_C: tl.constexpr,
):
    # program_id(0): n
    # program_id(1): oc tile
    # program_id(2): pooled output tile
    n = tl.program_id(0)
    oc_tile = tl.program_id(1)
    p_tile = tl.program_id(2)

    oc_offs = oc_tile * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    p_offs = p_tile * BLOCK_P + tl.arange(0, BLOCK_P)
    P_total = PH * PW
    p_mask = p_offs < P_total

    ph = p_offs // PW
    pw = p_offs % PW

    base_h = ph * POOL  # [BLOCK_P]
    base_w = pw * POOL

    # accumulator [BLOCK_OC, BLOCK_P]
    acc = tl.zeros((BLOCK_OC, BLOCK_P), dtype=tl.float32)

    # Loop over kh, kw, then ic (vectorized)
    ic_range = tl.arange(0, IC_C)  # [IC_C]

    for kh in tl.static_range(KH_C):
        for kw in tl.static_range(KW_C):
            # Sum over POOL x POOL
            for dy in tl.static_range(POOL):
                in_h = base_h + dy + kh  # [BLOCK_P]
                for dx in tl.static_range(POOL):
                    in_w = base_w + dx + kw  # [BLOCK_P]
                    # Load x[n, in_h, in_w, :] in NHWC: shape [BLOCK_P, IC_C]
                    x_offset = (n * stride_xn
                                + in_h[:, None] * stride_xh
                                + in_w[:, None] * stride_xw
                                + ic_range[None, :] * stride_xc)
                    x_vals = tl.load(x_ptr + x_offset,
                                     mask=p_mask[:, None],
                                     other=0.0)  # [BLOCK_P, IC_C]
                    # Load weight [BLOCK_OC, IC_C] for this kh,kw
                    # weight layout: [OC, IC, KH, KW]
                    w_offset = (oc_offs[:, None] * (IC * KH * KW)
                                + ic_range[None, :] * (KH * KW)
                                + kh * KW + kw)
                    w_vals = tl.load(w_ptr + w_offset,
                                     mask=oc_mask[:, None],
                                     other=0.0)  # [BLOCK_OC, IC_C]
                    # acc[oc, p] += sum_ic w_vals[oc, ic] * x_vals[p, ic]
                    # = w_vals @ x_vals.T
                    acc += tl.dot(w_vals, tl.trans(x_vals))

    # Add bias * POOL*POOL and divide by POOL*POOL => bias unchanged after avg
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    inv = 1.0 / (POOL * POOL)
    acc = acc * inv + bias[:, None]
    # Sigmoid
    acc = tl.sigmoid(acc)
    # Mask invalid pooled positions
    acc = tl.where(p_mask[None, :], acc, 0.0)
    # Sum over pooled positions => [BLOCK_OC]
    part = tl.sum(acc, axis=1)
    # Mask oc
    part = tl.where(oc_mask, part, 0.0)
    # Atomic add into out[n, oc_tile?] - we want per-(n) sum eventually
    # Add into out[n, oc_offs]
    tl.atomic_add(out_ptr + n * OC + oc_offs, part, mask=oc_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1
        POOL = self.pool_kernel_size
        PH = OH // POOL
        PW = OW // POOL

        # Convert x to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        stride_xn = x_nhwc.stride(0)
        stride_xh = x_nhwc.stride(1)
        stride_xw = x_nhwc.stride(2)
        stride_xc = x_nhwc.stride(3)

        out_partial = torch.zeros((N, OC), device=x.device, dtype=torch.float32)

        BLOCK_P = 64
        BLOCK_OC = 32
        P_total = PH * PW
        grid = (N, (OC + BLOCK_OC - 1) // BLOCK_OC, (P_total + BLOCK_P - 1) // BLOCK_P)

        # IC_C must be power of 2 >= IC. IC=8 ok.
        IC_C = max(IC, 16)
        # Pad weight if needed - actually IC=8, but tl.dot needs >= 16 typically
        # We need to ensure ic_range loads beyond IC return 0
        # But our kernel uses IC_C as constexpr arange size. If IC_C > IC,
        # we'd need masking. Let's just use IC=8 directly; tl.dot supports it
        # if BLOCK_OC * BLOCK_P large enough. Actually tl.dot requires inner dim >= 16.
        # We'll pad weight and input channel dim to IC_C=16.
        if IC < 16:
            IC_C = 16
            # Pad weight
            w_pad = torch.zeros((OC, IC_C, KH, KW), device=w.device, dtype=w.dtype)
            w_pad[:, :IC, :, :] = w
            w_use = w_pad.contiguous()
            # Pad x_nhwc on channel dim
            x_pad = torch.zeros((N, H, W, IC_C), device=x.device, dtype=x.dtype)
            x_pad[:, :, :, :IC] = x_nhwc
            x_use = x_pad.contiguous()
            stride_xn = x_use.stride(0)
            stride_xh = x_use.stride(1)
            stride_xw = x_use.stride(2)
            stride_xc = x_use.stride(3)
            IC_eff = IC_C
        else:
            IC_C = IC
            w_use = w
            x_use = x_nhwc
            IC_eff = IC

        conv_pool_sigmoid_sum_kernel[grid](
            x_use, w_use, b, out_partial,
            N, IC_eff, H, W,
            OC, KH, KW,
            OH, OW,
            PH, PW,
            stride_xn, stride_xh, stride_xw, stride_xc,
            POOL=POOL,
            BLOCK_P=BLOCK_P,
            BLOCK_OC=BLOCK_OC,
            KH_C=KH,
            KW_C=KW,
            IC_C=IC_C,
            num_warps=4,
            num_stages=2,
        )

        return out_partial.sum(dim=1)