import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_scatter_kernel(
    x_ptr, w_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # Output-tile based gather kernel
    # Grid: (N, OC/BLOCK_OC, OH*OW/BLOCK_SP)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (OH * OW)

    oh = sp_offs // OW
    ow = sp_offs % OW

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # ConvTranspose2d: out[n,oc,oh,ow] = sum_{ic,kh,kw} x[n,ic,ih,iw] * w[ic,oc,kh,kw]
    # where ih = (oh + PAD - kh) / STRIDE if divisible
    for kh in tl.static_range(0, KH):
        ih_num = oh + PAD - kh
        ih = ih_num // STRIDE
        ih_valid = (ih_num % STRIDE == 0) & (ih >= 0) & (ih < IH)
        for kw in tl.static_range(0, KW):
            iw_num = ow + PAD - kw
            iw = iw_num // STRIDE
            iw_valid = (iw_num % STRIDE == 0) & (iw >= 0) & (iw < IW)
            valid = ih_valid & iw_valid & sp_mask  # [BLOCK_SP]

            # Loop over IC tiles
            for ic_start in range(0, IC, BLOCK_IC):
                ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                ic_mask = ic_offs < IC

                # Load x tile: shape [BLOCK_IC, BLOCK_SP]
                x_off = (pid_n * (IC * IH * IW)
                         + ic_offs[:, None] * (IH * IW)
                         + ih[None, :] * IW
                         + iw[None, :])
                x_mask = ic_mask[:, None] & valid[None, :]
                x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                # Load w tile: shape [BLOCK_OC, BLOCK_IC]
                # w[ic, oc, kh, kw]
                w_off = (ic_offs[None, :] * (OC * KH * KW)
                         + oc_offs[:, None] * (KH * KW)
                         + kh * KW + kw)
                w_mask = oc_mask[:, None] & ic_mask[None, :]
                w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

                acc += tl.dot(w_tile, x_tile)

    # Store: out[n, oc, oh, ow]
    out_off = pid_n * (OC * OH * OW) + oc_offs[:, None] * (OH * OW) + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


@triton.jit
def epilogue_kernel(
    inp_ptr, bias_ptr, out_ptr,
    total_elements, sp_size, OC,
    INV_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elements

    x = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    # bias channel index
    oc_idx = (offs // sp_size) % OC
    b = tl.load(bias_ptr + oc_idx, mask=mask, other=0.0)
    x = x + b
    # clamp(0,1) -> *s -> clamp(0,1) -> /s simplifies to clamp(x,0,1/s) since 1*INV_SCALE = 1/s, but only when x*s>=1 saturates.
    # Actually: clamp(clamp(x,0,1)*s, 0,1) / s
    # Since clamp(x,0,1) is in [0,1], multiplied by s gives [0,s]. clamp to 1: [0,1]. Divide by s: [0, 1/s].
    # So result = clamp(clamp(x,0,1)*s, 0, 1)/s = min(clamp(x,0,1), 1/s) = clamp(x, 0, 1/s)
    x = tl.minimum(tl.maximum(x, 0.0), INV_SCALE)
    tl.store(out_ptr + offs, x, mask=mask)


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

        conv_out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 64
        BLOCK_SP = 64
        BLOCK_IC = 32
        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_SP))

        conv_transpose2d_scatter_kernel[grid](
            x, weight, conv_out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            self.stride, self.padding,
            BLOCK_OC, BLOCK_SP, BLOCK_IC,
            num_warps=4, num_stages=2,
        )

        out = torch.empty_like(conv_out)
        total = conv_out.numel()
        sp_size = OH * OW
        BLOCK = 1024
        grid2 = (triton.cdiv(total, BLOCK),)
        epilogue_kernel[grid2](
            conv_out, fused_bias, out,
            total, sp_size, OC,
            float(1.0 / self.scaling_factor),
            BLOCK,
            num_warps=4, num_stages=2,
        )
        return out