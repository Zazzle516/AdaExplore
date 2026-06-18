import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_P': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_P': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_P': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_P': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_P': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_P': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_P': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_P': 512}, num_warps=8, num_stages=2),
    ],
    key=['IC_VAL', 'KH_VAL', 'KW_VAL', 'POOL'],
)
@triton.jit
def conv_pool_sigmoid_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    PH, PW,
    POOL: tl.constexpr,
    IC_VAL: tl.constexpr,
    KH_VAL: tl.constexpr,
    KW_VAL: tl.constexpr,
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

    base_oh = ph * POOL
    base_ow = pw * POOL

    bias = tl.load(b_ptr + pid_oc)

    pooled = tl.zeros((BLOCK_P,), dtype=tl.float32)

    # Directly accumulate sum of conv outputs over the pool window
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
                        pooled += x_val * w_val

    pooled = pooled / (POOL * POOL) + bias
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

        total_p = PH * PW
        grid = lambda meta: (N, OC, (total_p + meta['BLOCK_P'] - 1) // meta['BLOCK_P'])

        conv_pool_sigmoid_sum_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            PH, PW,
            POOL=POOL,
            IC_VAL=IC,
            KH_VAL=KH,
            KW_VAL=KW,
        )

        return out