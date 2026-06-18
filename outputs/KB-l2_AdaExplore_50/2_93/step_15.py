import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose_fused_kernel(
    x_ptr,           # input  [N, IC, IH, IW]
    w_ptr,           # weight [IC, OC, KH, KW]
    bias_ptr,        # bias   [OC]
    out_ptr,         # output [N, OC, OH, OW]
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr,
    ADD_VAL: tl.constexpr,
    MUL_VAL: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)   # [BLOCK_OC]
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)   # [BLOCK_SP]

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (OH * OW)

    oh = sp_offs // OW
    ow = sp_offs % OW

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # For each kernel position, determine which input pixel contributes.
    # Output formula: oh = ih*STRIDE + kh  =>  ih = (oh - kh)/STRIDE if divisible
    for kh in tl.static_range(0, KH):
        ih_num = oh - kh   # [BLOCK_SP]
        ih = ih_num // STRIDE
        ih_valid = ((ih_num % STRIDE) == 0) & (ih >= 0) & (ih < IH)
        for kw in tl.static_range(0, KW):
            iw_num = ow - kw
            iw = iw_num // STRIDE
            iw_valid = ((iw_num % STRIDE) == 0) & (iw >= 0) & (iw < IW)
            valid = ih_valid & iw_valid & sp_mask  # [BLOCK_SP]

            # Load x[n, :, ih, iw] for each sp position -> shape [BLOCK_SP, IC]
            # We do a GEMM over IC.
            # x offset: n*IC*IH*IW + ic*IH*IW + ih*IW + iw
            # weight offset: ic*OC*KH*KW + oc*KH*KW + kh*KW + kw

            # Loop over IC in chunks
            BLOCK_IC: tl.constexpr = 64
            for ic_start in tl.static_range(0, IC, BLOCK_IC):
                ic_offs = ic_start + tl.arange(0, BLOCK_IC)  # [BLOCK_IC]
                ic_mask = ic_offs < IC

                # x_ptrs: [BLOCK_SP, BLOCK_IC]
                x_offset = (pid_n * IC * IH * IW
                            + ic_offs[None, :] * (IH * IW)
                            + ih[:, None] * IW
                            + iw[:, None])
                x_load_mask = valid[:, None] & ic_mask[None, :]
                x_vals = tl.load(x_ptr + x_offset, mask=x_load_mask, other=0.0)  # [BLOCK_SP, BLOCK_IC]

                # w_ptrs: [BLOCK_IC, BLOCK_OC]
                w_offset = (ic_offs[:, None] * (OC * KH * KW)
                            + oc_offs[None, :] * (KH * KW)
                            + kh * KW + kw)
                w_load_mask = ic_mask[:, None] & oc_mask[None, :]
                w_vals = tl.load(w_ptr + w_offset, mask=w_load_mask, other=0.0)  # [BLOCK_IC, BLOCK_OC]

                acc += tl.dot(x_vals, w_vals)

    # Add bias
    b = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    v = acc + b[None, :] + ADD_VAL

    # min(v, 0)
    v = tl.minimum(v, 0.0)
    # GELU exact
    inv_sqrt2 = 0.70710678118654752440
    gelu = 0.5 * v * (1.0 + tl.math.erf(v * inv_sqrt2))
    out = gelu * MUL_VAL

    # Store [BLOCK_SP, BLOCK_OC] into out[n, oc, oh, ow]
    out_offset = (pid_n * OC * OH * OW
                  + oc_offs[None, :] * (OH * OW)
                  + sp_offs[:, None])
    store_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_offset, out, mask=store_mask)


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
        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()  # [IC, OC, KH, KW]
        bias = self.conv_transpose.bias.contiguous()

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        S = self.stride

        OH = (IH - 1) * S + KH
        OW = (IW - 1) * S + KW

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 64
        BLOCK_SP = 64
        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_SP))

        conv_transpose_fused_kernel[grid](
            x, w, bias, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW, S,
            self.add_value, self.multiply_value,
            BLOCK_OC=BLOCK_OC,
            BLOCK_SP=BLOCK_SP,
            num_warps=4,
            num_stages=2,
        )
        return out