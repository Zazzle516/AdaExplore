import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_tanh_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    PH, PW,
    SUB1, SUB2,
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

    # We'll accumulate avg-pool sum over POOL*POOL output positions
    inv = 1.0 / (POOL * POOL)
    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    oc_mask = oc_offs < OC
    hw_mask = (ph < PH) & (pw < PW)

    # for each pooling cell, compute conv output then tanh
    for dh in tl.static_range(POOL):
        for dw in tl.static_range(POOL):
            oh = ph * POOL + dh
            ow = pw * POOL + dw
            # conv output at (n=pid_n, oc_offs, oh, ow)
            # init with bias
            bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
            conv_acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32) + bias[:, None]

            for ic in range(0, IC):
                for kh in tl.static_range(KH):
                    for kw in tl.static_range(KW):
                        ih = oh + kh
                        iw = ow + kw
                        # input pointer
                        x_off = ((pid_n * IC + ic) * IH + ih) * IW + iw
                        in_bounds = (ih < IH) & (iw < IW) & hw_mask
                        x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0)
                        # weight: (OC, IC, KH, KW)
                        w_off = ((oc_offs * IC + ic) * KH + kh) * KW + kw
                        w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                        conv_acc += w_val[:, None] * x_val[None, :]

            # subtract1, tanh, subtract2
            v = conv_acc - SUB1
            t = (tl.exp(v) - tl.exp(-v)) / (tl.exp(v) + tl.exp(-v))
            t = t - SUB2
            acc += t * inv

    # store
    out_off = ((pid_n * OC + oc_offs[:, None]) * PH + ph[None, :]) * PW + pw[None, :]
    out_mask = oc_mask[:, None] & hw_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


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

        BLOCK_OC = 32
        BLOCK_HW = 64

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(PH * PW, BLOCK_HW))

        conv_tanh_pool_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            PH, PW,
            self.subtract1_value, self.subtract2_value,
            KH, KW, POOL,
            BLOCK_OC, BLOCK_HW,
            num_warps=4, num_stages=2,
        )
        return out