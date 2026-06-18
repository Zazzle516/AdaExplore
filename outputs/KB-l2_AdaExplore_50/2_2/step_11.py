import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'BLOCK_IC': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_IC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 128, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64, 'BLOCK_IC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 256, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OH_OW_PAR', 'IC'],
)
@triton.jit
def conv_transpose2d_parity_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW, OH_OW_PAR,
    OH_PAR, OW_PAR,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    PHK_START: tl.constexpr,     # first kh value for this parity
    PWK_START: tl.constexpr,     # first kw value for this parity
    OH_OFF: tl.constexpr,        # output oh offset (oh_par_idx maps to oh = oh_par_idx*STRIDE + OH_OFF)
    OW_OFF: tl.constexpr,
    SCALE: tl.constexpr,
    INV_SCALE: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_sp = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_n = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < OH_OW_PAR

    # sp_offs indexes into the parity sub-grid of shape (OH_PAR, OW_PAR)
    oh_par_idx = sp_offs // OW_PAR
    ow_par_idx = sp_offs % OW_PAR

    oh = oh_par_idx * STRIDE + OH_OFF
    ow = ow_par_idx * STRIDE + OW_OFF

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # For this parity, only specific (kh, kw) taps are valid; iterate kh from PHK_START step STRIDE
    for kh in tl.static_range(PHK_START, KH, STRIDE):
        for kw in tl.static_range(PWK_START, KW, STRIDE):
            ih_num = oh + PAD - kh
            iw_num = ow + PAD - kw
            ih = ih_num // STRIDE
            iw = iw_num // STRIDE
            valid = (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW) & sp_mask

            for ic_start in range(0, IC, BLOCK_IC):
                ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                ic_mask = ic_offs < IC

                w_off = (ic_offs[None, :] * (OC * KH * KW)
                         + oc_offs[:, None] * (KH * KW)
                         + kh * KW + kw)
                w_mask = oc_mask[:, None] & ic_mask[None, :]
                w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

                x_off = (pid_n * (IC * IH * IW)
                         + ic_offs[:, None] * (IH * IW)
                         + ih[None, :] * IW + iw[None, :])
                x_mask = ic_mask[:, None] & valid[None, :]
                x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                acc += tl.dot(w_tile, x_tile, allow_tf32=True)

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[:, None]

    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    acc = acc * SCALE
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    acc = acc * INV_SCALE

    # Store: out[n, oc, oh, ow]; need real oh, ow indices
    out_sp_off = oh[None, :] * OW + ow[None, :]
    out_off = pid_n * (OC * OH * OW) + oc_offs[:, None] * (OH * OW) + out_sp_off
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
        S = self.stride
        P = self.padding
        OH = (IH - 1) * S - 2 * P + KH + self.output_padding
        OW = (IW - 1) * S - 2 * P + KW + self.output_padding

        fused_bias = (self.conv_transpose.bias + self.bias.view(-1)).contiguous()
        weight = self.conv_transpose.weight.contiguous()

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        scale = float(self.scaling_factor)
        inv_scale = 1.0 / scale

        # Launch one kernel per (oh%S, ow%S) parity
        for ph in range(S):
            for pw in range(S):
                # For oh ≡ ph (mod S), valid kh satisfies (oh + P - kh) % S == 0
                # => kh ≡ (ph + P) (mod S)
                phk_start = (ph + P) % S
                pwk_start = (pw + P) % S
                # Number of output rows/cols with oh%S==ph
                # oh ranges [0, OH), oh = ph + k*S for k in [0, ceil((OH-ph)/S))
                if ph >= OH:
                    OH_PAR = 0
                else:
                    OH_PAR = (OH - ph + S - 1) // S
                if pw >= OW:
                    OW_PAR = 0
                else:
                    OW_PAR = (OW - pw + S - 1) // S
                OH_OW_PAR = OH_PAR * OW_PAR
                if OH_OW_PAR == 0:
                    continue

                grid = lambda meta: (
                    triton.cdiv(OH_OW_PAR, meta['BLOCK_SP']),
                    triton.cdiv(OC, meta['BLOCK_OC']),
                    N,
                )

                conv_transpose2d_parity_kernel[grid](
                    x, weight, fused_bias, out,
                    N, IC, IH, IW,
                    OC, OH, OW, OH_OW_PAR,
                    OH_PAR, OW_PAR,
                    KH, KW,
                    S, P,
                    phk_start, pwk_start,
                    ph, pw,
                    scale, inv_scale,
                )

        return out