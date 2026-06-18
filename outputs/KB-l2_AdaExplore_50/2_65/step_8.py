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

    bias = tl.load(b_ptr + pid_oc)

    # Pooled accumulator over POOLxPOOL conv outputs
    acc = tl.zeros((BLOCK_P,), dtype=tl.float32)

    # The pool-window's conv outputs cover an input region of size:
    # (POOL + KH - 1) x (POOL + KW - 1), starting at input pos (ph*POOL, pw*POOL)
    # We can compute it as: for each (kh, kw, ic), accumulate w[oc,ic,kh,kw] * x[n,ic, ph*POOL + kh + dh, pw*POOL + kw + dw]
    # for dh,dw in [0,POOL). Equivalently swap loops: for each ic, for each input offset (ih_off, iw_off) in
    # [0, POOL+KH-1) x [0, POOL+KW-1), compute the count of (kh, dh) pairs that map to it -> a fixed coefficient.
    # But simpler: loop over (kh, kw) and (dh, dw); inside, do a single ic dot.

    base_h = ph * POOL  # [BLOCK_P]
    base_w = pw * POOL  # [BLOCK_P]

    for kh in tl.static_range(KH_VAL):
        for kw in tl.static_range(KW_VAL):
            for dh in tl.static_range(POOL):
                for dw in tl.static_range(POOL):
                    ih = base_h + dh + kh
                    iw = base_w + dw + kw
                    # dot over ic
                    s = tl.zeros((BLOCK_P,), dtype=tl.float32)
                    for ic in tl.static_range(IC_VAL):
                        x_off = ((pid_n * IC + ic) * IH + ih) * IW + iw
                        x_val = tl.load(x_ptr + x_off, mask=p_mask, other=0.0)
                        w_off = ((pid_oc * IC + ic) * KH_VAL + kh) * KW_VAL + kw
                        w_val = tl.load(w_ptr + w_off)
                        s += x_val * w_val
                    acc += s

    # add bias * POOL*POOL (since each of POOL*POOL conv outputs adds bias)
    acc += bias * (POOL * POOL)

    # average pool
    pooled = acc / (POOL * POOL)
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

        BLOCK_P = 64
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
            num_warps=4,
            num_stages=2,
        )

        return out