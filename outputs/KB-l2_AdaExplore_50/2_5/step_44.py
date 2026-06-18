import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _conv_transpose_bias_tanh_kernel(
    x_ptr,        # [N, IC, IH, IW]  channels_last? no, NCHW contiguous
    w_ptr,        # [IC, OC, KH, KW]
    b_ptr,        # [OC]
    bias_ptr,     # [OC]  (the subtract bias)
    out_ptr,      # [N, OC, OH, OW]
    N, IC, OC,
    IH, IW, OH, OW,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # Grid: (N, ceil(OC/BLOCK_OC), ceil(OH*OW/BLOCK_SP))
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    offs_sp = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]

    mask_oc = offs_oc < OC
    mask_sp = offs_sp < (OH * OW)

    oh = offs_sp // OW  # [BLOCK_SP]
    ow = offs_sp % OW   # [BLOCK_SP]

    # stride=2, padding=1, kernel=4, output_padding=1 => OH = 2*IH, OW = 2*IW
    # For each output pixel:
    # ih_num = oh + pad - kh; needs ih_num % stride == 0; ih = ih_num/stride; 0 <= ih < IH
    # With kernel=4, pad=1, stride=2:
    #   parity of (oh - kh + 1) must be even => kh has parity (oh+1) mod 2
    #   So kh in {0,2} if (oh+1)%2==0 i.e oh odd? Let's compute:
    #   (oh + 1 - kh) % 2 == 0 => kh % 2 == (oh+1) % 2
    # We'll iterate kh in {0,2} or {1,3} based on parity of oh.
    # Similarly for kw.

    # Pre-compute kh start parity
    kh_par = (oh + 1) % 2  # [BLOCK_SP], values 0 or 1
    kw_par = (ow + 1) % 2  # [BLOCK_SP]

    # Accumulator
    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # Loop over IC in blocks
    # For the 4 (kh, kw) pairs that contribute:
    # kh_vals = [kh_par, kh_par + 2]
    # kw_vals = [kw_par, kw_par + 2]

    for kh_idx in tl.static_range(0, 2):
        # kh = kh_par + kh_idx*2 ; per-element value
        kh = kh_par + kh_idx * 2  # [BLOCK_SP]
        ih = (oh + 1 - kh) // 2   # [BLOCK_SP]
        valid_h = (ih >= 0) & (ih < IH)

        for kw_idx in tl.static_range(0, 2):
            kw = kw_par + kw_idx * 2  # [BLOCK_SP]
            iw = (ow + 1 - kw) // 2
            valid_w = (iw >= 0) & (iw < IW)
            valid_hw = valid_h & valid_w  # [BLOCK_SP]

            # Loop over IC blocks
            for ic_start in range(0, IC, BLOCK_IC):
                offs_ic = ic_start + tl.arange(0, BLOCK_IC)  # [BLOCK_IC]
                mask_ic = offs_ic < IC

                # Load input: x[n, ic, ih, iw] for each (sp, ic)
                # input offset = n*IC*IH*IW + ic*IH*IW + ih*IW + iw
                ih_safe = tl.where(valid_hw, ih, 0)
                iw_safe = tl.where(valid_hw, iw, 0)
                x_offs = (pid_n * IC * IH * IW
                          + offs_ic[None, :] * (IH * IW)
                          + ih_safe[:, None] * IW
                          + iw_safe[:, None])  # [BLOCK_SP, BLOCK_IC]
                x_mask = (valid_hw[:, None]) & mask_ic[None, :]
                x_vals = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)  # [BLOCK_SP, BLOCK_IC]

                # Load weight: w[ic, oc, kh, kw]
                # But kh, kw are per-sp scalars... they vary per sp.
                # However within this static iter, kh_idx and kw_idx are fixed,
                # so kh = kh_par + kh_idx*2 is per-sp. That means weight depends on sp.
                # We need w[ic, oc, kh[sp], kw[sp]] — different per sp.
                # offset = ic*OC*KH*KW + oc*KH*KW + kh*KW + kw
                # KH=KW=4
                w_offs = (offs_ic[:, None, None] * (OC * 16)
                          + offs_oc[None, None, :] * 16
                          + kh[None, :, None] * 4
                          + kw[None, :, None])  # [BLOCK_IC, BLOCK_SP, BLOCK_OC]
                w_mask = mask_ic[:, None, None] & mask_oc[None, None, :] & valid_hw[None, :, None]
                # This 3D weight load is large. Alternative: since kh,kw vary per sp,
                # we can't use tl.dot easily. Let's do explicit reduction.

                w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)  # [BLOCK_IC, BLOCK_SP, BLOCK_OC]

                # acc[sp, oc] += sum_ic x[sp, ic] * w[ic, sp, oc]
                # = sum over ic of x[sp,ic,None] * w[ic,sp,oc] -> reduce axis ic
                # reshape: x [BLOCK_SP, BLOCK_IC, 1], w [BLOCK_IC, BLOCK_SP, BLOCK_OC] -> need transpose
                # Use elementwise multiply with broadcast:
                # x[sp, ic] -> [sp, ic, 1]; w[ic, sp, oc] -> permute to [sp, ic, oc]
                # reduce over ic
                prod = x_vals[:, :, None] * tl.trans(w_vals, (1, 0, 2))  # [BLOCK_SP, BLOCK_IC, BLOCK_OC]
                acc += tl.sum(prod, axis=1)

    # Add conv bias
    cb = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)  # [BLOCK_OC]
    acc += cb[None, :]

    # Subtract user bias
    sb = tl.load(bias_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc -= sb[None, :]

    # tanh
    e2x = tl.exp(2.0 * acc)
    out = (e2x - 1.0) / (e2x + 1.0)

    # Store: out[n, oc, oh, ow]
    out_offs = (pid_n * OC * OH * OW
                + offs_oc[None, :] * (OH * OW)
                + offs_sp[:, None])
    out_mask = mask_sp[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_offs, out, mask=out_mask)


def conv_transpose_bias_tanh(x, weight, conv_bias, sub_bias):
    N, IC, IH, IW = x.shape
    _, OC, KH, KW = weight.shape
    OH = 2 * IH
    OW = 2 * IW

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 64
    BLOCK_SP = 64
    BLOCK_IC = 16

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_SP))
    _conv_transpose_bias_tanh_kernel[grid](
        x, weight, conv_bias, sub_bias, out,
        N, IC, OC, IH, IW, OH, OW,
        BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP, BLOCK_IC=BLOCK_IC,
        num_warps=4, num_stages=2,
    )
    return out


from triton.language.extra import libdevice as _libdev


@triton.jit
def _bias_tanh_nchw_kernel(
    x_ptr, b_ptr, out_ptr,
    C, HW, TOTAL,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL

    pid_c = (offs // HW) % C
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + pid_c, mask=mask, other=0.0)
    y = x - b
    out = _libdev.tanh(y)
    tl.store(out_ptr + offs, out, mask=mask)


def fused_bias_tanh_nchw(x, bias):
    N, C, H, W = x.shape
    HW = H * W
    TOTAL = N * C * HW
    out = torch.empty_like(x)
    BLOCK = 4096
    grid = ((TOTAL + BLOCK - 1) // BLOCK,)
    _bias_tanh_nchw_kernel[grid](x, bias, out, C, HW, TOTAL, BLOCK=BLOCK, num_warps=4, num_stages=2)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        torch.backends.cudnn.benchmark = True
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size

    def forward(self, x):
        # Use cuDNN ConvTranspose2d (highly tuned), then fused bias-subtract + tanh.
        x = self.conv_transpose(x)
        b = self.bias.view(-1)
        return fused_bias_tanh_nchw(x, b)