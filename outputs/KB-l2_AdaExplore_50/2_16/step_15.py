import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    ADD_VALUE: tl.constexpr,
    SCALE: tl.constexpr,
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oh = sp_offs // OW
    ow = sp_offs % OW

    sp_mask = sp_offs < (OH * OW)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    n_x_base = pid_n * IC * IH * IW
    n_o_base = pid_n * OC * OH * OW

    for kh in tl.static_range(0, KH):
        ih_num = oh + PAD - kh
        ih = ih_num // STRIDE
        ih_valid = (ih_num >= 0) & ((ih_num % STRIDE) == 0) & (ih < IH) & (ih >= 0)
        for kw in tl.static_range(0, KW):
            iw_num = ow + PAD - kw
            iw = iw_num // STRIDE
            iw_valid = (iw_num >= 0) & ((iw_num % STRIDE) == 0) & (iw < IW) & (iw >= 0)
            valid = ih_valid & iw_valid & sp_mask

            x_spatial = ih * IW + iw
            w_oc_kk = oc_offs * (KH * KW) + (kh * KW + kw)

            for ic_start in range(0, IC, BLOCK_IC):
                ic_block = ic_start + tl.arange(0, BLOCK_IC)
                ic_mask = ic_block < IC

                x_ptrs = x_ptr + n_x_base + ic_block[None, :] * (IH * IW) + x_spatial[:, None]
                x_load_mask = valid[:, None] & ic_mask[None, :]
                x_vals = tl.load(x_ptrs, mask=x_load_mask, other=0.0)

                w_ptrs = w_ptr + ic_block[:, None] * (OC * KH * KW) + w_oc_kk[None, :]
                w_load_mask = ic_mask[:, None] & oc_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_load_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals, allow_tf32=True)

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[None, :]

    # Mish via tanh(softplus(x)) = (e^x - e^-x) / (e^x + e^-x + 2*e^-? ...) -> use stable form
    # Use: softplus(x) = max(x,0) + log1p(exp(-|x|))
    abs_x = tl.abs(acc)
    sp_val = tl.maximum(acc, 0.0) + tl.log(1.0 + tl.exp(-abs_x))
    # tanh(sp) = (exp(2*sp)-1)/(exp(2*sp)+1), but sp can be large. Use stable:
    # tanh(s) where s>=0 always since softplus>=0
    e_neg_2s = tl.exp(-2.0 * sp_val)
    th = (1.0 - e_neg_2s) / (1.0 + e_neg_2s)
    y = acc * th
    y = y + ADD_VALUE
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)
    y = y * SCALE

    out_ptrs = out_ptr + n_o_base + oc_offs[None, :] * (OH * OW) + sp_offs[:, None]
    store_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptrs, y, mask=store_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = float(add_value)
        self.scale = float(scale)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()
        b = self.conv_transpose.bias.contiguous()

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        STRIDE = self.stride
        PAD = self.padding
        OUT_PAD = self.output_padding

        OH = (IH - 1) * STRIDE - 2 * PAD + KH + OUT_PAD
        OW = (IW - 1) * STRIDE - 2 * PAD + KW + OUT_PAD

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 64
        BLOCK_SP = 64
        BLOCK_IC = 32

        grid = (N, (OC + BLOCK_OC - 1) // BLOCK_OC, (OH * OW + BLOCK_SP - 1) // BLOCK_SP)

        conv_transpose2d_fused_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            self.add_value, self.scale,
            STRIDE, PAD, KH, KW,
            BLOCK_OC=BLOCK_OC,
            BLOCK_SP=BLOCK_SP,
            BLOCK_IC=BLOCK_IC,
            num_warps=4,
            num_stages=3,
        )

        return out