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
    PH, PW,  # pooled output spatial = OH//pool, OW//pool
    POOL: tl.constexpr,
    BLOCK_P: tl.constexpr,
    KH_C: tl.constexpr,
    KW_C: tl.constexpr,
    IC_C: tl.constexpr,
):
    # Each program: one (n, oc) pair, processes BLOCK_P pooled output positions
    pid = tl.program_id(0)
    pid_p = tl.program_id(1)

    n = pid // OC
    oc = pid % OC

    p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    P_total = PH * PW
    p_mask = p_offs < P_total

    ph = p_offs // PW
    pw = p_offs % PW

    # Load bias
    bias = tl.load(b_ptr + oc)

    # For each pooled position, we need to sum over POOL*POOL conv outputs
    # Each conv output is sum over IC, KH, KW of x * w
    # Pooled value = (1/POOL^2) * sum over pool window of conv outputs
    # Sum over pool window of conv outputs = sum_{dy,dx} sum_{ic,kh,kw} x[n,ic, ph*POOL+dy+kh, pw*POOL+dx+kw] * w[oc,ic,kh,kw] + POOL^2 * bias

    acc = tl.zeros((BLOCK_P,), dtype=tl.float32)

    # Loop over ic, kh, kw
    for ic in tl.static_range(IC_C):
        for kh in tl.static_range(KH_C):
            for kw in tl.static_range(KW_C):
                # weight scalar
                w_val = tl.load(w_ptr + oc * IC * KH * KW + ic * KH * KW + kh * KW + kw)
                # Sum over pool window
                for dy in tl.static_range(POOL):
                    for dx in tl.static_range(POOL):
                        in_h = ph * POOL + dy + kh
                        in_w = pw * POOL + dx + kw
                        in_idx = n * IC * H * W + ic * H * W + in_h * W + in_w
                        valid = p_mask & (in_h < H) & (in_w < W)
                        x_val = tl.load(x_ptr + in_idx, mask=valid, other=0.0)
                        acc = acc + x_val * w_val

    # Add bias contribution: POOL*POOL * bias per pooled position
    acc = acc + bias * (POOL * POOL)
    # Average
    acc = acc / (POOL * POOL)
    # Sigmoid
    acc = tl.sigmoid(acc)
    # Mask invalid
    acc = tl.where(p_mask, acc, 0.0)
    # Sum over BLOCK_P
    block_sum = tl.sum(acc, axis=0)
    # Atomic add into out[n]
    tl.atomic_add(out_ptr + n, block_sum)


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

        out = torch.zeros(N, device=x.device, dtype=torch.float32)

        BLOCK_P = 128
        P_total = PH * PW
        grid = (N * OC, (P_total + BLOCK_P - 1) // BLOCK_P)

        conv_pool_sigmoid_sum_kernel[grid](
            x, w, b, out,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW,
            PH, PW,
            POOL=POOL,
            BLOCK_P=BLOCK_P,
            KH_C=KH,
            KW_C=KW,
            IC_C=IC,
            num_warps=4,
        )

        return out