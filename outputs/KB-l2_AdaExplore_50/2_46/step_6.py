import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
    ],
    key=['IC', 'OC', 'PH', 'PW', 'KH', 'KW'],
)
@triton.jit
def conv_tanh_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    PH, PW,
    SUB1: tl.constexpr, SUB2: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    x: (N, IC, IH, IW)
    w: (OC, IC, KH, KW)
    b: (OC,)
    out: (N, OC, PH, PW)

    For each pooled output position, we compute a POOL*POOL window of conv outputs
    and average them after tanh. The conv is implemented as a GEMM over K=IC*KH*KW
    using tl.dot for tensor-core throughput.
    """
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)        # [BLOCK_OC]
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)        # [BLOCK_HW]

    ph = hw_offs // PW
    pw = hw_offs % PW

    oc_mask = oc_offs < OC
    hw_mask = (ph < PH) & (pw < PW)

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)    # [BLOCK_OC]

    inv = 1.0 / (POOL * POOL)
    pooled_acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    K = IC * KH * KW
    k_range = tl.arange(0, BLOCK_K)                              # [BLOCK_K]

    for dh in tl.static_range(POOL):
        for dw in tl.static_range(POOL):
            oh = ph * POOL + dh    # [BLOCK_HW]
            ow = pw * POOL + dw    # [BLOCK_HW]

            acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

            for k0 in range(0, K, BLOCK_K):
                k_idx = k0 + k_range                              # [BLOCK_K]
                k_mask = k_idx < K

                ic = k_idx // (KH * KW)
                rem = k_idx % (KH * KW)
                kh = rem // KW
                kw = rem % KW

                # Weight: [BLOCK_OC, BLOCK_K]
                w_off = ((oc_offs[:, None] * IC + ic[None, :]) * KH + kh[None, :]) * KW + kw[None, :]
                w_mask = oc_mask[:, None] & k_mask[None, :]
                w_val = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

                # Input: [BLOCK_K, BLOCK_HW]
                ih = oh[None, :] + kh[:, None]    # [BLOCK_K, BLOCK_HW]
                iw = ow[None, :] + kw[:, None]    # [BLOCK_K, BLOCK_HW]
                x_off = ((pid_n * IC + ic[:, None]) * IH + ih) * IW + iw
                x_mask = k_mask[:, None] & hw_mask[None, :] & (ih < IH) & (iw < IW)
                x_val = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                acc += tl.dot(w_val, x_val, allow_tf32=True)

            v = acc + bias[:, None] - SUB1
            t = 2.0 * tl.sigmoid(2.0 * v) - 1.0
            t = t - SUB2
            pooled_acc += t * inv

    out_off = ((pid_n * OC + oc_offs[:, None]) * PH + ph[None, :]) * PW + pw[None, :]
    out_mask = oc_mask[:, None] & hw_mask[None, :]
    tl.store(out_ptr + out_off, pooled_acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = float(subtract1_value)
        self.subtract2_value = float(subtract2_value)
        self.kernel_size_pool = int(kernel_size_pool)
        self.kernel_size = int(kernel_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1
        POOL = self.kernel_size_pool
        PH = OH // POOL
        PW = OW // POOL

        out = torch.empty((N, OC, PH, PW), device=x.device, dtype=x.dtype)

        BLOCK_K = 64

        grid = lambda META: (N, triton.cdiv(OC, META['BLOCK_OC']), triton.cdiv(PH * PW, META['BLOCK_HW']))

        conv_tanh_pool_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            PH, PW,
            self.subtract1_value, self.subtract2_value,
            KH, KW, POOL,
            BLOCK_K=BLOCK_K,
        )
        return out