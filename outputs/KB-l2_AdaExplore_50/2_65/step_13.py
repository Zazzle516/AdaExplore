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
    OH, OW,
    PH, PW,
    POOL: tl.constexpr,
    BLOCK_P: tl.constexpr,
    KH_C: tl.constexpr,
    KW_C: tl.constexpr,
    IC_C: tl.constexpr,
):
    # Each program handles one (n, oc) pair, loops over all pooled positions
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid % OC

    P_total = PH * PW

    bias = tl.load(b_ptr + oc)

    # Pre-load weights for this oc into registers (IC * KH * KW values)
    # Use a 3D static layout
    total_acc = tl.zeros((1,), dtype=tl.float32)

    num_tiles = (P_total + BLOCK_P - 1) // BLOCK_P

    for tile_idx in range(0, num_tiles):
        p_offs = tile_idx * BLOCK_P + tl.arange(0, BLOCK_P)
        p_mask = p_offs < P_total
        ph = p_offs // PW
        pw = p_offs % PW

        acc = tl.zeros((BLOCK_P,), dtype=tl.float32)

        for ic in tl.static_range(IC_C):
            for kh in tl.static_range(KH_C):
                for kw in tl.static_range(KW_C):
                    w_val = tl.load(w_ptr + oc * IC_C * KH_C * KW_C + ic * KH_C * KW_C + kh * KW_C + kw)
                    for dy in tl.static_range(POOL):
                        for dx in tl.static_range(POOL):
                            in_h = ph * POOL + dy + kh
                            in_w = pw * POOL + dx + kw
                            in_idx = n * IC_C * H * W + ic * H * W + in_h * W + in_w
                            x_val = tl.load(x_ptr + in_idx, mask=p_mask, other=0.0)
                            acc = acc + x_val * w_val

        acc = acc + bias * (POOL * POOL)
        acc = acc * (1.0 / (POOL * POOL))
        acc = tl.sigmoid(acc)
        acc = tl.where(p_mask, acc, 0.0)
        total_acc = total_acc + tl.sum(acc, axis=0)

    tl.store(out_ptr + n * OC + oc, tl.sum(total_acc, axis=0))


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

        buf = torch.empty((N, OC), device=x.device, dtype=torch.float32)

        BLOCK_P = 256
        grid = (N * OC,)

        conv_pool_sigmoid_sum_kernel[grid](
            x, w, b, buf,
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

        return buf.sum(dim=1)