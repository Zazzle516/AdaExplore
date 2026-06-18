import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_pool_sigmoid_sum_kernel(
    conv_ptr,     # [N, OC, OH, OW]
    out_ptr,      # [N]
    OC, OH, OW,
    PH, PW,
    POOL: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)

    P_total = PH * PW
    base = pid_n * OC * OH * OW + pid_oc * OH * OW

    inv = 1.0 / (POOL * POOL)

    acc_sum = tl.zeros((BLOCK_P,), dtype=tl.float32)

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
    tl.atomic_add(out_ptr + pid_n, total)


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
        conv_out = self.conv(x)
        conv_out = conv_out.contiguous()

        N, OC, OH, OW = conv_out.shape
        POOL = self.pool_kernel_size
        PH = OH // POOL
        PW = OW // POOL

        out = torch.zeros(N, device=x.device, dtype=torch.float32)

        BLOCK_P = 512
        grid = (N, OC)

        fused_pool_sigmoid_sum_kernel[grid](
            conv_out, out,
            OC, OH, OW,
            PH, PW,
            POOL=POOL,
            BLOCK_P=BLOCK_P,
            num_warps=8,
            num_stages=3,
        )

        return out