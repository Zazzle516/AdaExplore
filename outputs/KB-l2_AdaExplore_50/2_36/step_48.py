import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose_min_sum_gelu_kernel(
    x_ptr, w_ptr, b_ptr, bias_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_w = tl.program_id(1)

    w_start = pid_w * BLOCK_W
    ow_offs = w_start + tl.arange(0, BLOCK_W)
    ow_mask = ow_offs < OW

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    ic_offs = tl.arange(0, BLOCK_IC)
    ic_mask = ic_offs < IC

    b_vec = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    b_vec_masked = tl.where(oc_mask, b_vec, float('inf'))

    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    x_base = pid_n * IC * IH * IW

    # Precompute per-kw iw indices/validity (independent of oh)
    # We'll compute them in the inner kw loop each oh anyway since cheap.

    for oh in range(0, OH):
        conv_tile = tl.broadcast_to(b_vec_masked[:, None], (BLOCK_OC, BLOCK_W))

        for kh in tl.static_range(0, KH):
            ih_num = oh + PH - kh
            ih = ih_num // SH
            ih_valid = (ih_num >= 0) & (ih_num % SH == 0) & (ih >= 0) & (ih < IH)

            for kw in tl.static_range(0, KW):
                iw_num = ow_offs + PW - kw
                iw = iw_num // SW
                iw_valid = (iw_num >= 0) & (iw_num % SW == 0) & (iw >= 0) & (iw < IW)
                valid = ih_valid & iw_valid & ow_mask

                x_offs = (x_base
                          + ic_offs[:, None] * (IH * IW)
                          + ih * IW
                          + iw[None, :])
                x_m = ic_mask[:, None] & valid[None, :]
                x_vals = tl.load(x_ptr + x_offs, mask=x_m, other=0.0)

                w_offs = (ic_offs[:, None] * (OC * KH * KW)
                          + oc_offs[None, :] * (KH * KW)
                          + kh * KW
                          + kw)
                w_m = ic_mask[:, None] & oc_mask[None, :]
                w_vals = tl.load(w_ptr + w_offs, mask=w_m, other=0.0)

                w_t = tl.trans(w_vals)
                conv_tile += tl.dot(w_t, x_vals)

        min_vec = tl.min(conv_tile, axis=0)
        acc += min_vec

    inv_sqrt2 = 0.7071067811865475
    gelu_out = 0.5 * acc * (1.0 + tl.math.erf(acc * inv_sqrt2))

    bias_val = tl.load(bias_ptr)
    result = gelu_out + bias_val

    out_offs = pid_n * OW + ow_offs
    tl.store(out_ptr + out_offs, result, mask=ow_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
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
        SH = SW = self.stride
        PH = PW = self.padding
        OP = self.output_padding
        OH = (IH - 1) * SH - 2 * PH + KH + OP
        OW = (IW - 1) * SW - 2 * PW + KW + OP

        weight = self.conv_transpose.weight.contiguous()
        conv_bias = self.conv_transpose.bias.contiguous() if self.conv_transpose.bias is not None else torch.zeros(OC, device=x.device, dtype=x.dtype)
        bias = self.bias.contiguous()

        out = torch.empty((N, 1, 1, OW), device=x.device, dtype=x.dtype)

        BLOCK_W = 128
        BLOCK_OC = max(triton.next_power_of_2(OC), 16)
        BLOCK_IC = max(triton.next_power_of_2(IC), 16)
        grid = (N, triton.cdiv(OW, BLOCK_W))

        conv_transpose_min_sum_gelu_kernel[grid](
            x, weight, conv_bias, bias, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW, SH, SW, PH, PW,
            BLOCK_W=BLOCK_W,
            BLOCK_OC=BLOCK_OC,
            BLOCK_IC=BLOCK_IC,
            num_warps=8,
            num_stages=3,
        )

        return out