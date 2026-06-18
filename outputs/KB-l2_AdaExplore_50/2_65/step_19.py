import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_pool_sigmoid_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    PH, PW,
    POOL: tl.constexpr,
    BLOCK_P: tl.constexpr,
    IC_VAL: tl.constexpr,
    KH_VAL: tl.constexpr,
    KW_VAL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_p = tl.program_id(2)

    p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    total_p = PH * PW
    p_mask = p_offs < total_p

    ph = p_offs // PW
    pw = p_offs % PW

    # base coordinates of conv output corresponding to top-left of pool window
    base_oh = ph * POOL  # [BLOCK_P]
    base_ow = pw * POOL  # [BLOCK_P]

    bias = tl.load(b_ptr + pid_oc)

    # 2D accumulator: one conv accumulator per pool tap
    POOL2: tl.constexpr = POOL * POOL
    acc_conv = tl.zeros((POOL2, BLOCK_P), dtype=tl.float32)

    # Loop order: (ic, kh, kw) outer, load weight once, accumulate into all pool taps
    for ic in tl.static_range(IC_VAL):
        for kh in tl.static_range(KH_VAL):
            for kw in tl.static_range(KW_VAL):
                w_off = ((pid_oc * IC_VAL + ic) * KH_VAL + kh) * KW_VAL + kw
                w_val = tl.load(w_ptr + w_off)
                for kh_idx in tl.static_range(POOL):
                    for kw_idx in tl.static_range(POOL):
                        ih = base_oh + kh_idx + kh
                        iw = base_ow + kw_idx + kw
                        x_off = ((pid_n * IC + ic) * IH + ih) * IW + iw
                        x_val = tl.load(x_ptr + x_off, mask=p_mask, other=0.0)
                        tap = kh_idx * POOL + kw_idx
                        # accumulate per-tap conv result
                        # build a one-hot mask along tap axis to update only that row
                        tap_range = tl.arange(0, POOL2)
                        tap_mask = (tap_range == tap)[:, None]
                        acc_conv += tl.where(tap_mask, x_val[None, :] * w_val, 0.0)

    # add bias to every tap
    acc_conv += bias
    # avg pool sum across taps then divide
    pooled = tl.sum(acc_conv, axis=0) / (POOL * POOL)
    sig = 1.0 / (1.0 + tl.exp(-pooled))
    sig = tl.where(p_mask, sig, 0.0)
    partial = tl.sum(sig, axis=0)

    tl.atomic_add(out_ptr + pid_n, partial)


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

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1
        POOL = self.pool_kernel_size
        PH = OH // POOL
        PW = OW // POOL

        out = torch.zeros(N, device=x.device, dtype=torch.float32)

        BLOCK_P = 128
        total_p = PH * PW
        grid = (N, OC, (total_p + BLOCK_P - 1) // BLOCK_P)

        conv_pool_sigmoid_sum_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            PH, PW,
            POOL=POOL,
            BLOCK_P=BLOCK_P,
            IC_VAL=IC,
            KH_VAL=KH,
            KW_VAL=KW,
            num_warps=8,
            num_stages=2,
        )

        return out