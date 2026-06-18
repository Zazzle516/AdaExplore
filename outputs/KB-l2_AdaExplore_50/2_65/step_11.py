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

    acc = tl.zeros((BLOCK_P,), dtype=tl.float32)

    base_h = ph * POOL
    base_w = pw * POOL
    x_n_base = n * IC * H * W

    # Loop over ic, kh, kw
    for ic in tl.static_range(IC_C):
        x_ic_base = x_n_base + ic * H * W
        w_ic_base = oc * IC * KH * KW + ic * KH * KW
        for kh in tl.static_range(KH_C):
            for kw in tl.static_range(KW_C):
                w_val = tl.load(w_ptr + w_ic_base + kh * KW + kw)
                # Sum over pool window
                for dy in tl.static_range(POOL):
                    in_h = base_h + dy + kh
                    row_base = x_ic_base + in_h * W
                    for dx in tl.static_range(POOL):
                        in_w = base_w + dx + kw
                        in_idx = row_base + in_w
                        x_val = tl.load(x_ptr + in_idx, mask=p_mask, other=0.0)
                        acc = acc + x_val * w_val

    # Add bias contribution: POOL*POOL * bias per pooled position
    acc = acc + bias * (POOL * POOL)
    # Average
    inv = 1.0 / (POOL * POOL)
    acc = acc * inv
    # Sigmoid
    acc = tl.sigmoid(acc)
    # Mask invalid
    acc = tl.where(p_mask, acc, 0.0)
    # Sum over BLOCK_P
    block_sum = tl.sum(acc, axis=0)
    # Atomic add into out[n, oc] (much less contention than out[n])
    tl.atomic_add(out_ptr + n * OC + oc, block_sum)


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

        out_partial = torch.zeros((N, OC), device=x.device, dtype=torch.float32)

        BLOCK_P = 256
        P_total = PH * PW
        grid = (N * OC, (P_total + BLOCK_P - 1) // BLOCK_P)

        conv_pool_sigmoid_sum_kernel[grid](
            x, w, b, out_partial,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW,
            PH, PW,
            POOL=POOL,
            BLOCK_P=BLOCK_P,
            KH_C=KH,
            KW_C=KW,
            IC_C=IC,
            num_warps=8,
            num_stages=2,
        )

        return out_partial.sum(dim=1)