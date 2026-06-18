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
    OH, OW,  # conv output spatial
    PH, PW,  # pooled output spatial
    POOL: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_P: tl.constexpr,
    IC_C: tl.constexpr,
    KH_C: tl.constexpr,
    KW_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_p = tl.program_id(2)

    oc_off = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    p_off = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)

    oc_mask = oc_off < OC
    p_mask = p_off < (PH * PW)

    ph = p_off // PW
    pw = p_off % PW

    # accumulator: sigmoid(avg_pool(conv)) summed over BLOCK_OC x BLOCK_P
    # we will accumulate per (oc, p) the pooled value
    pooled = tl.zeros((BLOCK_OC, BLOCK_P), dtype=tl.float32)

    # Iterate over pooling window
    for ki in tl.static_range(0, POOL):
        for kj in tl.static_range(0, POOL):
            oh = ph * POOL + ki  # conv output row
            ow = pw * POOL + kj  # conv output col

            # Compute conv output value at (oh, ow) for each oc in BLOCK_OC
            # acc shape: [BLOCK_OC, BLOCK_P]
            acc = tl.zeros((BLOCK_OC, BLOCK_P), dtype=tl.float32)
            for ic in tl.static_range(0, IC_C):
                for kh in tl.static_range(0, KH_C):
                    for kw in tl.static_range(0, KW_C):
                        ih = oh + kh  # input row (no padding)
                        iw = ow + kw  # input col
                        # Load input: [BLOCK_P], indexed by (ih, iw) per p
                        in_idx = pid_n * IC * H * W + ic * H * W + ih * W + iw
                        in_mask = p_mask & (ih < H) & (iw < W)
                        x_val = tl.load(x_ptr + in_idx, mask=in_mask, other=0.0)  # [BLOCK_P]

                        # Load weight: [BLOCK_OC]
                        w_idx = oc_off * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                        w_val = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                        acc += w_val[:, None] * x_val[None, :]

            # Add bias
            bias = tl.load(b_ptr + oc_off, mask=oc_mask, other=0.0)
            acc = acc + bias[:, None]
            pooled += acc

    pooled = pooled / (POOL * POOL)
    sig = tl.sigmoid(pooled)
    sig = tl.where(oc_mask[:, None] & p_mask[None, :], sig, 0.0)

    # Reduce over oc and p for this program -> partial sum, atomically add to out[n]
    s = tl.sum(tl.sum(sig, axis=1), axis=0)
    tl.atomic_add(out_ptr + pid_n, s)


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
        POOL = self.pool_kernel_size

        OH = H - KH + 1
        OW = W - KW + 1
        PH = OH // POOL
        PW = OW // POOL

        out = torch.zeros(N, device=x.device, dtype=torch.float32)

        BLOCK_OC = 16
        BLOCK_P = 64

        grid = (
            N,
            (OC + BLOCK_OC - 1) // BLOCK_OC,
            (PH * PW + BLOCK_P - 1) // BLOCK_P,
        )

        conv_pool_sigmoid_sum_kernel[grid](
            x, w, b, out,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW,
            PH, PW,
            POOL=POOL,
            BLOCK_OC=BLOCK_OC,
            BLOCK_P=BLOCK_P,
            IC_C=IC,
            KH_C=KH,
            KW_C=KW,
            num_warps=4,
            num_stages=2,
        )

        return out