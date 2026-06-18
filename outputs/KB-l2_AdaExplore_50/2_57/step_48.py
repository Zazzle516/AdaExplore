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
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OH', 'OW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv_relu_hardswish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, OH, OW, K_TOTAL,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_sp = tl.program_id(0)  # over N * OH * OW (fused)
    pid_oc = tl.program_id(1)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    OHW = OH * OW
    total_sp = N * OHW

    n_idx = sp_offs // OHW
    sp_local = sp_offs % OHW
    oh = sp_local // OW
    ow = sp_local % OW

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < total_sp

    KHW: tl.constexpr = KH * KW
    HW = H * W

    # Precompute spatial input offset [BLOCK_SP]: n*IC*HW + oh*W + ow
    sp_in_off = n_idx * (IC * HW) + oh * W + ow

    # Precompute K offsets (static, BLOCK_K = padded K_TOTAL)
    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < K_TOTAL
    ic = k_offs // KHW
    rem = k_offs % KHW
    kh = rem // KW
    kw = rem % KW
    k_in_off = ic * HW + kh * W + kw

    # weight tile [BLOCK_OC, BLOCK_K]
    w_off = oc_offs[:, None] * K_TOTAL + k_offs[None, :]
    w_mask = oc_mask[:, None] & k_mask[None, :]
    w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

    # input tile [BLOCK_K, BLOCK_SP]
    x_off = k_in_off[:, None] + sp_in_off[None, :]
    x_m = k_mask[:, None] & sp_mask[None, :]
    x_tile = tl.load(x_ptr + x_off, mask=x_m, other=0.0)

    acc = tl.dot(w_tile, x_tile)

    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b_val[:, None]

    acc = tl.maximum(acc, 0.0)
    hs = tl.minimum(tl.maximum((acc + 3.0) / 6.0, 0.0), 1.0)
    out = acc * hs

    out_off = (n_idx[None, :] * (OC * OHW)
               + oc_offs[:, None] * OHW
               + sp_local[None, :])
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
        # Pad K to next power-of-two for static unroll
        BLOCK_K = 1
        while BLOCK_K < K_TOTAL:
            BLOCK_K *= 2

        grid = lambda META: (
            triton.cdiv(N * OH * OW, META['BLOCK_SP']),
            triton.cdiv(OC, META['BLOCK_OC']),
        )

        conv_relu_hardswish_kernel[grid](
            x, w, b, out,
            N, IC, H, W,
            OC, OH, OW, K_TOTAL,
            KH, KW, BLOCK_K,
        )
        return out