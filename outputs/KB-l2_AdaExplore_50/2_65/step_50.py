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
    BLOCK_PH: tl.constexpr,
    BLOCK_PW: tl.constexpr,
    OC_C: tl.constexpr,
    IC_C: tl.constexpr,
    KH_C: tl.constexpr,
    KW_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_ph = tl.program_id(1)
    pid_pw = tl.program_id(2)

    ph_base = pid_ph * BLOCK_PH
    pw_base = pid_pw * BLOCK_PW

    OH_TILE: tl.constexpr = BLOCK_PH * POOL
    OW_TILE: tl.constexpr = BLOCK_PW * POOL
    M: tl.constexpr = OH_TILE * OW_TILE

    oc_range = tl.arange(0, OC_C)
    bias = tl.load(b_ptr + oc_range)

    inv_pool2 = 1.0 / (POOL * POOL)

    conv_acc = tl.zeros([M, OC_C], dtype=tl.float32)
    conv_acc += bias[None, :]

    out_idx = tl.arange(0, M)
    out_r = out_idx // OW_TILE
    out_c = out_idx % OW_TILE

    oh_global = ph_base * POOL + out_r
    ow_global = pw_base * POOL + out_c

    for ic in tl.static_range(0, IC_C):
        for kh in tl.static_range(0, KH_C):
            for kw in tl.static_range(0, KW_C):
                ih = oh_global + kh
                iw = ow_global + kw
                x_off = ((pid_n * IC + ic) * H + ih) * W + iw
                x_mask = (ih < H) & (iw < W)
                xv = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)
                w_off = ((oc_range * IC + ic) * KH + kh) * KW + kw
                wv = tl.load(w_ptr + w_off)
                conv_acc += xv[:, None] * wv[None, :]

    conv_5d = tl.reshape(conv_acc, [BLOCK_PH, POOL, BLOCK_PW, POOL, OC_C])
    pooled = tl.sum(tl.sum(conv_5d, axis=3), axis=1)  # [BLOCK_PH, BLOCK_PW, OC_C]
    pooled = pooled * inv_pool2

    p_idx = tl.arange(0, BLOCK_PH)
    q_idx = tl.arange(0, BLOCK_PW)
    ph_global = ph_base + p_idx
    pw_global = pw_base + q_idx
    valid = (ph_global[:, None] < PH) & (pw_global[None, :] < PW)

    sig = tl.sigmoid(pooled)
    sig = tl.where(valid[:, :, None], sig, 0.0)
    s = tl.sum(tl.sum(tl.sum(sig, axis=2), axis=1), axis=0)

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

        if PH * POOL != OH or PW * POOL != OW:
            y = self.conv(x)
            y = self.avg_pool(y)
            y = torch.sigmoid(y)
            return torch.sum(y, dim=[1, 2, 3])

        out = torch.zeros(N, device=x.device, dtype=torch.float32)

        BLOCK_PH = 4
        BLOCK_PW = 8
        n_ph = (PH + BLOCK_PH - 1) // BLOCK_PH
        n_pw = (PW + BLOCK_PW - 1) // BLOCK_PW

        grid = (N, n_ph, n_pw)
        conv_pool_sigmoid_sum_kernel[grid](
            x, w, b, out,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW, PH, PW,
            POOL=POOL,
            BLOCK_PH=BLOCK_PH,
            BLOCK_PW=BLOCK_PW,
            OC_C=OC,
            IC_C=IC,
            KH_C=KH,
            KW_C=KW,
            num_warps=8,
            num_stages=2,
        )
        return out