import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Scatter-add based ConvTranspose2d:
# For each input position (n, ic, ih, iw), compute contributions to output:
#   oh = ih*stride - pad + kh
#   ow = iw*stride - pad + kw
# For stride=2, pad=1, kernel=3, output_pad=1: OH = 2*IH, OW = 2*IW
# Each input element contributes to KH*KW output positions, accumulating across IC.
#
# Strategy: 
#   - one program per (n, OC tile, output spatial tile)
#   - gather inputs via the inverse mapping (the "gather" formulation) which
#     for stride=2 has each (oh, ow) mapping to exactly one (ih, iw) per (kh, kw)
#   - Do GEMM over IC dimension
#   - NHWC-like output store: keep output as NCHW but tile is (OC, SP)
#   - Use TMA-style large tiles, BLOCK_OC=64, BLOCK_SP=128


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64,  'BLOCK_SP': 128, 'BLOCK_IC': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64,  'BLOCK_SP': 128, 'BLOCK_IC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64,  'BLOCK_IC': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64,  'BLOCK_IC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64,  'BLOCK_SP': 64,  'BLOCK_IC': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 32,  'BLOCK_SP': 128, 'BLOCK_IC': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64,  'BLOCK_SP': 256, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'IC', 'OH', 'OW'],
)
@triton.jit
def conv_transpose2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW, SP,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    SCALE: tl.constexpr,
    INV_SCALE: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_sp = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_n  = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < SP

    oh = sp_offs // OW
    ow = sp_offs - oh * OW

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    x_batch_off = pid_n * (IC * IH * IW)

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih_num = oh + PAD - kh
            iw_num = ow + PAD - kw
            ih = ih_num // STRIDE
            iw = iw_num // STRIDE
            valid = (ih_num % STRIDE == 0) & (iw_num % STRIDE == 0)
            valid = valid & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW) & sp_mask

            x_spatial_off = ih * IW + iw  # [BLOCK_SP]

            for ic_start in range(0, IC, BLOCK_IC):
                ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                ic_mask = ic_offs < IC

                # x[n, ic, ih, iw]  -> [BLOCK_IC, BLOCK_SP]
                x_addrs = x_batch_off + ic_offs[:, None] * (IH * IW) + x_spatial_off[None, :]
                x_mask = ic_mask[:, None] & valid[None, :]
                x_tile = tl.load(x_ptr + x_addrs, mask=x_mask, other=0.0)

                # w[ic, oc, kh, kw] -> [BLOCK_OC, BLOCK_IC]
                w_addrs = ic_offs[None, :] * (OC * KH * KW) + oc_offs[:, None] * (KH * KW) + kh * KW + kw
                w_mask = oc_mask[:, None] & ic_mask[None, :]
                w_tile = tl.load(w_ptr + w_addrs, mask=w_mask, other=0.0)

                acc += tl.dot(w_tile, x_tile, allow_tf32=True)

    # Bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[:, None]

    # clamp(0,1) -> *scale -> clamp(0,1) -> /scale
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    acc = acc * SCALE
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    acc = acc * INV_SCALE

    out_off = pid_n * (OC * OH * OW) + oc_offs[:, None] * SP + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding,
                                                  output_padding=output_padding)
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
        SP = OH * OW

        grid = lambda meta: (triton.cdiv(SP, meta['BLOCK_SP']),
                             triton.cdiv(OC, meta['BLOCK_OC']),
                             N)

        conv_transpose2d_kernel[grid](
            x, weight, fused_bias, out,
            N, IC, IH, IW,
            OC, OH, OW, SP,
            KH, KW,
            self.stride, self.padding,
            float(self.scaling_factor),
            float(1.0 / self.scaling_factor),
        )
        return out