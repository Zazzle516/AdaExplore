import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_fused_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr,
    ADD_VALUE: tl.constexpr,
    MUL_VALUE: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # grid: (N, ceil(OC/BLOCK_OC), ceil(OH*OW/BLOCK_SP))
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    offs_sp = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]

    oh = offs_sp // OW
    ow = offs_sp % OW

    mask_oc = offs_oc < OC
    mask_sp = offs_sp < (OH * OW)

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # For each kh, kw, find ih, iw such that ih*STRIDE + kh = oh (no padding)
    # ih = (oh - kh) / STRIDE, valid when (oh - kh) >= 0 and (oh - kh) % STRIDE == 0 and ih < IH
    for kh in tl.static_range(0, KH):
        oh_kh = oh - kh
        ih = oh_kh // STRIDE
        valid_h = (oh_kh >= 0) & ((oh_kh % STRIDE) == 0) & (ih < IH) & (ih >= 0)
        for kw in tl.static_range(0, KW):
            ow_kw = ow - kw
            iw = ow_kw // STRIDE
            valid_w = (ow_kw >= 0) & ((ow_kw % STRIDE) == 0) & (iw < IW) & (iw >= 0)
            valid = valid_h & valid_w  # [BLOCK_SP]

            ih_safe = tl.where(valid, ih, 0)
            iw_safe = tl.where(valid, iw, 0)

            # input offset: x[n, ic, ih, iw] = n*IC*IH*IW + ic*IH*IW + ih*IW + iw
            x_base = pid_n * IC * IH * IW + ih_safe * IW + iw_safe  # [BLOCK_SP]

            # weight: w[ic, oc, kh, kw], layout [IC, OC, KH, KW]
            w_base = (kh * KW + kw) + offs_oc * KH * KW  # [BLOCK_OC]

            # Loop over IC in blocks
            for ic_start in range(0, IC, BLOCK_IC):
                offs_ic = ic_start + tl.arange(0, BLOCK_IC)  # [BLOCK_IC]
                mask_ic = offs_ic < IC

                # Load x: shape [BLOCK_SP, BLOCK_IC]
                x_ptrs = x_ptr + x_base[:, None] + offs_ic[None, :] * IH * IW
                x_mask = valid[:, None] & mask_ic[None, :] & mask_sp[:, None]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

                # Load w: shape [BLOCK_IC, BLOCK_OC]
                w_ptrs = w_ptr + offs_ic[:, None] * OC * KH * KW + w_base[None, :]
                w_mask = mask_ic[:, None] & mask_oc[None, :]
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals)

    # Add bias
    bias = tl.load(bias_ptr + offs_oc, mask=mask_oc, other=0.0)  # [BLOCK_OC]
    v = acc + bias[None, :] + ADD_VALUE
    v = tl.minimum(v, 0.0)
    inv_sqrt2 = 0.70710678118654752440
    gelu = 0.5 * v * (1.0 + tl.math.erf(v * inv_sqrt2))
    out = gelu * MUL_VALUE

    # Store: out[n, oc, oh, ow]
    out_base = pid_n * OC * OH * OW + offs_oc[None, :] * OH * OW + offs_sp[:, None]
    out_mask = mask_sp[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_base, out, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = float(add_value)
        self.multiply_value = float(multiply_value)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv_transpose.weight.contiguous()
        bias = self.conv_transpose.bias.contiguous()

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        STRIDE = self.stride

        OH = (IH - 1) * STRIDE + KH
        OW = (IW - 1) * STRIDE + KW

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 64
        BLOCK_SP = 64
        BLOCK_IC = 32

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_SP))

        conv_transpose_fused_kernel[grid](
            x, w, bias, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH=KH, KW=KW,
            STRIDE=STRIDE,
            ADD_VALUE=self.add_value,
            MUL_VALUE=self.multiply_value,
            BLOCK_OC=BLOCK_OC,
            BLOCK_SP=BLOCK_SP,
            BLOCK_IC=BLOCK_IC,
            num_warps=4,
            num_stages=2,
        )
        return out