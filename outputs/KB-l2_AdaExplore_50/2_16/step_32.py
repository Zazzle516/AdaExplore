import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    add_value: tl.constexpr, scale: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]

    oh = sp_offs // OW
    ow = sp_offs % OW

    sp_mask = sp_offs < (OH * OW)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # For each (kh, kw), find input pixel that contributes
    # ih = (oh + PAD - kh) / STRIDE iff (oh + PAD - kh) % STRIDE == 0
    for kh in tl.static_range(KH):
        ih_num = oh + PAD - kh  # [BLOCK_SP]
        ih = ih_num // STRIDE
        ih_valid = (ih_num % STRIDE == 0) & (ih >= 0) & (ih < IH)
        for kw in tl.static_range(KW):
            iw_num = ow + PAD - kw
            iw = iw_num // STRIDE
            iw_valid = (iw_num % STRIDE == 0) & (iw >= 0) & (iw < IW)
            valid = ih_valid & iw_valid & sp_mask  # [BLOCK_SP]

            # Loop over IC
            # x: [N, IC, IH, IW], w: [IC, OC, KH, KW]
            # Use blocked IC reduction
            BLOCK_IC: tl.constexpr = 16
            for ic_start in tl.static_range(0, 64, BLOCK_IC):
                ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                # load x[n, ic, ih, iw]: shape [BLOCK_SP, BLOCK_IC]
                x_ptrs = (x_ptr
                          + pid_n * IC * IH * IW
                          + ic_offs[None, :] * IH * IW
                          + ih[:, None] * IW
                          + iw[:, None])
                x_vals = tl.load(x_ptrs, mask=valid[:, None], other=0.0)

                # load w[ic, oc, kh, kw]: shape [BLOCK_IC, BLOCK_OC]
                w_ptrs = (w_ptr
                          + ic_offs[:, None] * (OC * KH * KW)
                          + oc_offs[None, :] * (KH * KW)
                          + kh * KW + kw)
                w_vals = tl.load(w_ptrs, mask=oc_mask[None, :], other=0.0)

                acc += tl.dot(x_vals, w_vals)

    # Add bias
    b = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    y = acc + b[None, :]

    # Mish: y * tanh(softplus(y))
    sp_val = tl.log(1.0 + tl.exp(y))
    e2 = tl.exp(2.0 * sp_val)
    th = (e2 - 1.0) / (e2 + 1.0)
    y = y * th
    y = y + add_value
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)
    y = y * scale

    # Store: out[n, oc, oh, ow]
    out_ptrs = (out_ptr
                + pid_n * OC * OH * OW
                + oc_offs[None, :] * (OH * OW)
                + sp_offs[:, None])
    out_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptrs, y, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        S = self.stride
        P = self.padding
        OP = self.output_padding
        OH = (IH - 1) * S - 2 * P + KH + OP
        OW = (IW - 1) * S - 2 * P + KW + OP

        weight = self.conv_transpose.weight.contiguous()  # [IC, OC, KH, KW]
        bias = self.conv_transpose.bias.contiguous()

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_SP = 64

        grid = (N, (OC + BLOCK_OC - 1) // BLOCK_OC, (OH * OW + BLOCK_SP - 1) // BLOCK_SP)

        conv_transpose_fused_kernel[grid](
            x, weight, bias, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            S, P,
            float(self.add_value), float(self.scale),
            BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
            num_warps=4, num_stages=2,
        )
        return out