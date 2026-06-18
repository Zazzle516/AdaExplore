import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 256}, num_warps=8, num_stages=3),
    ],
    key=['OC', 'OH', 'OW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv_relu_hardswish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, OH, OW,
    K_TOTAL: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_sp = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_n = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oh = sp_offs // OW
    ow = sp_offs % OW

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (OH * OW)

    HW = H * W
    x_batch_off = pid_n * (IC * HW)

    sp_in_off = oh * W + ow

    KHW: tl.constexpr = KH * KW

    # Full K (IC*KH*KW = 72) in one tile
    k_offs = tl.arange(0, K_TOTAL)
    ic = k_offs // KHW
    rem = k_offs % KHW
    kh = rem // KW
    kw = rem % KW
    k_in_off = ic * HW + kh * W + kw

    # weight tile [BLOCK_OC, K_TOTAL]
    w_off = oc_offs[:, None] * K_TOTAL + k_offs[None, :]
    w_mask = oc_mask[:, None]
    w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

    # input tile [K_TOTAL, BLOCK_SP]
    x_off = x_batch_off + k_in_off[:, None] + sp_in_off[None, :]
    x_m = sp_mask[None, :]
    x_tile = tl.load(x_ptr + x_off, mask=x_m, other=0.0)

    acc = tl.dot(w_tile, x_tile)

    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b_val[:, None]

    acc = tl.maximum(acc, 0.0)
    hs = tl.minimum((acc + 3.0) / 6.0, 1.0)
    out = acc * hs

    out_off = (pid_n * OC * OH * OW
               + oc_offs[:, None] * (OH * OW)
               + sp_offs[None, :])
    mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()
        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        K_TOTAL = IC * KH * KW

        grid = lambda META: (
            triton.cdiv(OH * OW, META['BLOCK_SP']),
            triton.cdiv(OC, META['BLOCK_OC']),
            N,
        )

        conv_relu_hardswish_kernel[grid](
            x, w, b, out,
            N, IC, H, W,
            OC, OH, OW,
            K_TOTAL,
            KH, KW,
        )
        return out