import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_pool_sigmoid_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, KH, KW,
    OH, OW,
    PH, PW,  # pooled H, W
    POOL: tl.constexpr,
    BLOCK_P: tl.constexpr,
    IC_VAL: tl.constexpr,
    KH_VAL: tl.constexpr,
    KW_VAL: tl.constexpr,
):
    # one program per (n, oc, pool-tile)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_p = tl.program_id(2)

    p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    total_p = PH * PW
    p_mask = p_offs < total_p

    ph = p_offs // PW
    pw = p_offs % PW

    # pooled position corresponds to conv-output window starting at (ph*POOL, pw*POOL),
    # of size POOL x POOL
    acc = tl.zeros((BLOCK_P,), dtype=tl.float32)

    bias = tl.load(b_ptr + pid_oc)

    # Loop over the pool window
    for kh_idx in tl.static_range(POOL):
        for kw_idx in tl.static_range(POOL):
            oh = ph * POOL + kh_idx  # conv output coord
            ow = pw * POOL + kw_idx

            # Compute conv output at (n=pid_n, oc=pid_oc, oh, ow)
            # = sum over ic, kh, kw of x[n, ic, oh+kh, ow+kw] * w[oc, ic, kh, kw]
            conv_val = tl.zeros((BLOCK_P,), dtype=tl.float32)

            for ic in tl.static_range(IC_VAL):
                for kh in tl.static_range(KH_VAL):
                    for kw in tl.static_range(KW_VAL):
                        ih = oh + kh
                        iw = ow + kw
                        # mask: valid pooled position
                        x_off = ((pid_n * IC + ic) * IH + ih) * IW + iw
                        x_val = tl.load(x_ptr + x_off, mask=p_mask, other=0.0)
                        w_off = ((pid_oc * IC + ic) * KH_VAL + kh) * KW_VAL + kw
                        w_val = tl.load(w_ptr + w_off)
                        conv_val += x_val * w_val

            conv_val += bias
            acc += conv_val

    # average pool: divide by POOL*POOL
    pooled = acc / (POOL * POOL)
    # sigmoid
    sig = 1.0 / (1.0 + tl.exp(-pooled))
    # mask invalid positions
    sig = tl.where(p_mask, sig, 0.0)
    # reduce over BLOCK_P
    partial = tl.sum(sig, axis=0)

    # atomic add to out[pid_n]
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
            OC, KH, KW,
            OH, OW,
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