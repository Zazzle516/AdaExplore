import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Strategy:
# 1. Use cuDNN for the conv (it's heavily optimized for fp32 on RTX 4090).
# 2. Custom Triton kernel that does avg_pool + sigmoid + sum in one fused pass,
#    one program per N, reducing over (OC, PH, PW) without atomics.
# Conv output is materialized (satisfies safety contract), but the downstream
# reduction chain runs as a single fused kernel without writing intermediates.


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_P': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_P': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_P': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_P': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_P': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_P': 1024}, num_warps=8, num_stages=2),
    ],
    key=['OH', 'OW', 'POOL', 'OC'],
)
@triton.jit
def fused_pool_sigmoid_sum_kernel(
    conv_ptr,     # [N, OC, OH, OW]
    out_ptr,      # [N]
    N, OC, OH, OW,
    PH, PW,
    POOL: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid_n = tl.program_id(0)

    P_total = PH * PW
    inv = 1.0 / (POOL * POOL)

    acc_sum = tl.zeros((BLOCK_P,), dtype=tl.float32)

    for oc in range(0, OC):
        base = pid_n * OC * OH * OW + oc * OH * OW
        for p_start in range(0, P_total, BLOCK_P):
            p_off = p_start + tl.arange(0, BLOCK_P)
            p_mask = p_off < P_total

            ph = p_off // PW
            pw = p_off % PW

            pooled = tl.zeros((BLOCK_P,), dtype=tl.float32)
            for ki in tl.static_range(0, POOL):
                for kj in tl.static_range(0, POOL):
                    oh = ph * POOL + ki
                    ow = pw * POOL + kj
                    idx = base + oh * OW + ow
                    v = tl.load(conv_ptr + idx, mask=p_mask, other=0.0)
                    pooled += v

            pooled = pooled * inv
            sig = tl.sigmoid(pooled)
            sig = tl.where(p_mask, sig, 0.0)
            acc_sum += sig

    total = tl.sum(acc_sum, axis=0)
    tl.store(out_ptr + pid_n, total)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        torch.backends.cudnn.benchmark = True
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # Use channels_last for faster cuDNN algo on Ampere/Ada
        self.conv = self.conv.to(memory_format=torch.channels_last)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = x.cuda().contiguous(memory_format=torch.channels_last)

        # cuDNN conv with channels_last layout
        conv_out = self.conv(x)  # [N, OC, OH, OW] channels_last

        # Convert to NCHW contiguous for our pool kernel (still cheap vs conv)
        conv_out = conv_out.contiguous()

        N, OC, OH, OW = conv_out.shape
        POOL = self.pool_kernel_size
        PH = OH // POOL
        PW = OW // POOL

        out = torch.empty(N, device=x.device, dtype=torch.float32)

        grid = (N,)

        fused_pool_sigmoid_sum_kernel[grid](
            conv_out, out,
            N, OC, OH, OW,
            PH, PW,
            POOL=POOL,
        )

        return out