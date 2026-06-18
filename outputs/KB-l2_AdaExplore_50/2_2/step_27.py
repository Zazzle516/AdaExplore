import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OW', 'IC'],
)
@triton.jit
def conv_transpose2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    INV_SCALE: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_sp = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_n = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    OH_OW = OH * OW
    oc_mask = oc_offs < OC
    sp_mask = sp_offs < OH_OW

    oh = sp_offs // OW
    ow = sp_offs % OW

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # Preload x base offsets
    x_n_base = pid_n * (IC * IH * IW)

    # Unroll over kh,kw with static range
    for kh in tl.static_range(0, KH):
        ih_num = oh + PAD - kh
        ih = ih_num // STRIDE
        ih_ok = (ih_num % STRIDE == 0) & (ih >= 0) & (ih < IH)
        for kw in tl.static_range(0, KW):
            iw_num = ow + PAD - kw
            iw = iw_num // STRIDE
            iw_ok = (iw_num % STRIDE == 0) & (iw >= 0) & (iw < IW)
            valid = ih_ok & iw_ok & sp_mask  # [BLOCK_SP]

            # Single-tile IC (IC=64 fits)
            ic_offs = tl.arange(0, IC)

            # W tile [BLOCK_OC, IC]: w[ic, oc, kh, kw]
            w_off = (ic_offs[None, :] * (OC * KH * KW)
                     + oc_offs[:, None] * (KH * KW)
                     + kh * KW + kw)
            w_tile = tl.load(w_ptr + w_off, mask=oc_mask[:, None], other=0.0)

            # X tile [IC, BLOCK_SP]: x[n, ic, ih, iw]
            x_off = (x_n_base
                     + ic_offs[:, None] * (IH * IW)
                     + ih[None, :] * IW + iw[None, :])
            x_tile = tl.load(x_ptr + x_off, mask=valid[None, :], other=0.0)

            acc += tl.dot(w_tile, x_tile, allow_tf32=True)

    # Bias + fused clamp/scale/clamp/div -> clamp(x+b, 0, 1/s)
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[:, None]
    acc = tl.minimum(tl.maximum(acc, 0.0), INV_SCALE)

    out_off = pid_n * (OC * OH_OW) + oc_offs[:, None] * OH_OW + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous()
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = (IH - 1) * self.stride - 2 * self.padding + KH + self.output_padding
        OW = (IW - 1) * self.stride - 2 * self.padding + KW + self.output_padding

        fused_bias = (self.conv_transpose.bias + self.bias.view(-1)).contiguous()
        weight = self.conv_transpose.weight.contiguous()

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)
        OH_OW = OH * OW

        grid = lambda meta: (
            triton.cdiv(OH_OW, meta['BLOCK_SP']),
            triton.cdiv(OC, meta['BLOCK_OC']),
            N,
        )

        conv_transpose2d_kernel[grid](
            x, weight, fused_bias, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            self.stride, self.padding,
            float(1.0 / self.scaling_factor),
        )
        return out