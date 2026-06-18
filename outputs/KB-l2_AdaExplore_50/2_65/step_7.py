import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_P': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_P': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_P': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_P': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_P': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_P': 512}, num_warps=8, num_stages=2),
    ],
    key=['H', 'W', 'OC', 'IC_C'],
)
@triton.jit
def conv_pool_sigmoid_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, KH, KW,
    OH, OW,           # conv output dims
    PH, PW,           # pooled dims
    PARTIAL_STRIDE_N, PARTIAL_STRIDE_OC,
    POOL: tl.constexpr,
    KH_C: tl.constexpr,
    KW_C: tl.constexpr,
    IC_C: tl.constexpr,
    BLOCK_P: tl.constexpr,
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

    inv_pool2 = 1.0 / (POOL * POOL)

    pooled_acc = tl.zeros([BLOCK_P], dtype=tl.float32)

    for dy in tl.static_range(0, POOL):
        for dx in tl.static_range(0, POOL):
            oh = ph * POOL + dy
            ow = pw * POOL + dx

            conv_val = tl.zeros([BLOCK_P], dtype=tl.float32) + bias

            for ic in tl.static_range(0, IC_C):
                for kh in tl.static_range(0, KH_C):
                    for kw in tl.static_range(0, KW_C):
                        ih = oh + kh
                        iw = ow + kw
                        x_off = ((pid_n * IC + ic) * H + ih) * W + iw
                        xv = tl.load(x_ptr + x_off, mask=p_mask, other=0.0)
                        w_off = ((pid_oc * IC + ic) * KH + kh) * KW + kw
                        wv = tl.load(w_ptr + w_off)
                        conv_val += xv * wv
            pooled_acc += conv_val * inv_pool2

    sig = tl.sigmoid(pooled_acc)
    sig = tl.where(p_mask, sig, 0.0)
    s = tl.sum(sig, axis=0)

    out_off = pid_n * PARTIAL_STRIDE_N + pid_oc * PARTIAL_STRIDE_OC + pid_p
    tl.store(out_ptr + out_off, s)


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

        # max tiles based on smallest BLOCK_P in autotune
        MIN_BLOCK_P = 128
        max_tiles = (PH * PW + MIN_BLOCK_P - 1) // MIN_BLOCK_P
        partial = torch.zeros((N, OC, max_tiles), device=x.device, dtype=torch.float32)

        def grid(meta):
            BP = meta['BLOCK_P']
            return (N, OC, (PH * PW + BP - 1) // BP)

        conv_pool_sigmoid_sum_kernel[grid](
            x, w, b, partial,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW, PH, PW,
            OC * max_tiles, max_tiles,
            POOL=POOL,
            KH_C=KH,
            KW_C=KW,
            IC_C=IC,
        )

        # Only first n_tiles_used per autotuned config contribute; unused entries
        # are uninitialized. To be safe, sum based on the actual used tile count
        # by zeroing the partial buffer beforehand. Instead, we'll initialize to 0.
        return partial.sum(dim=[1, 2])