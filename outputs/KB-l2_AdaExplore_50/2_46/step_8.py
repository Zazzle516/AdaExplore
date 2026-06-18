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
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=4, num_stages=3),
    ],
    key=['IC', 'OC', 'PH', 'PW'],
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
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    ph = hw_offs // PW
    pw = hw_offs % PW

    inv = 1.0 / (POOL * POOL)

    oc_mask = oc_offs < OC
    hw_mask = (ph < PH) & (pw < PW)

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    # base output spatial position (top-left of pool window)
    oh_base = ph * POOL  # [BLOCK_HW]
    ow_base = pw * POOL  # [BLOCK_HW]

    pool_acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    # iterate over POOL x POOL conv outputs and fuse pool average
    for dh in tl.static_range(POOL):
        for dw in tl.static_range(POOL):
            oh = oh_base + dh
            ow = ow_base + dw

            conv_acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32) + bias[:, None]

            # loop over kh, kw outside ic so we can reuse w_val and x_val patterns
            for kh in tl.static_range(KH):
                for kw in tl.static_range(KW):
                    ih = oh + kh
                    iw = ow + kw
                    in_bounds = hw_mask  # ih,iw are always in range since OH+KH-1 = IH
                    for ic in range(0, IC):
                        x_off = ((pid_n * IC + ic) * IH + ih) * IW + iw
                        x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0)
                        w_off = ((oc_offs * IC + ic) * KH + kh) * KW + kw
                        w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                        conv_acc += w_val[:, None] * x_val[None, :]

            v = conv_acc - SUB1
            t = 2.0 * tl.sigmoid(2.0 * v) - 1.0
            t = t - SUB2
            pool_acc += t * inv

    out_off = ((pid_n * OC + oc_offs[:, None]) * PH + ph[None, :]) * PW + pw[None, :]
    out_mask = oc_mask[:, None] & hw_mask[None, :]
    tl.store(out_ptr + out_off, pool_acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=3),
    ],
    key=['IC', 'OC', 'PH', 'PW', 'BLOCK_IC'],
)
@triton.jit
def conv_tanh_pool_gemm_kernel(
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
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    ph = hw_offs // PW
    pw = hw_offs % PW

    inv = 1.0 / (POOL * POOL)

    oc_mask = oc_offs < OC
    hw_mask = (ph < PH) & (pw < PW)

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    pool_acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    ic_offs = tl.arange(0, BLOCK_IC)

    for dh in tl.static_range(POOL):
        for dw in tl.static_range(POOL):
            oh = ph * POOL + dh
            ow = pw * POOL + dw

            conv_acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32) + bias[:, None]

            for kh in tl.static_range(KH):
                for kw in tl.static_range(KW):
                    ih = oh + kh
                    iw = ow + kw
                    # GEMM along IC: load weight tile [BLOCK_OC, BLOCK_IC], x tile [BLOCK_IC, BLOCK_HW]
                    for ic_start in range(0, IC, BLOCK_IC):
                        ic_idx = ic_start + ic_offs
                        ic_mask = ic_idx < IC
                        # weight: [OC, IC, KH, KW]
                        w_off = ((oc_offs[:, None] * IC + ic_idx[None, :]) * KH + kh) * KW + kw
                        w_tile = tl.load(w_ptr + w_off,
                                         mask=oc_mask[:, None] & ic_mask[None, :],
                                         other=0.0)
                        # x: [N, IC, IH, IW]
                        x_off = ((pid_n * IC + ic_idx[:, None]) * IH + ih[None, :]) * IW + iw[None, :]
                        x_tile = tl.load(x_ptr + x_off,
                                         mask=ic_mask[:, None] & hw_mask[None, :],
                                         other=0.0)
                        conv_acc += tl.dot(w_tile, x_tile, allow_tf32=True)

            v = conv_acc - SUB1
            t = 2.0 * tl.sigmoid(2.0 * v) - 1.0
            t = t - SUB2
            pool_acc += t * inv

    out_off = ((pid_n * OC + oc_offs[:, None]) * PH + ph[None, :]) * PW + pw[None, :]
    out_mask = oc_mask[:, None] & hw_mask[None, :]
    tl.store(out_ptr + out_off, pool_acc, mask=out_mask)


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

        # GEMM-based kernel using tl.dot
        BLOCK_IC = 32 if IC >= 32 else 16
        grid = lambda META: (N, triton.cdiv(OC, META['BLOCK_OC']), triton.cdiv(PH * PW, META['BLOCK_HW']))

        conv_tanh_pool_gemm_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            PH, PW,
            self.subtract1_value, self.subtract2_value,
            KH, KW, POOL,
            BLOCK_IC=BLOCK_IC,
        )
        return out