import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
    ],
    key=['N', 'OH', 'OW', 'IC', 'OC'],
)
@triton.jit
def conv_transpose_fused_kernel(
    x_ptr,        # [N, IC, IH, IW]
    w_ptr,        # [IC, OC, KH, KW]
    b_ptr,        # [OC]
    out_ptr,      # [N, OC, OH, OW]
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    ADD_VAL: tl.constexpr, SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # M dimension = N * OH * OW (output spatial * batch)
    # N dimension = OC
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    M = N * OH * OW
    m_mask = offs_m < M
    n_mask = offs_n < OC

    # decompose offs_m -> (n_idx, oh, ow)
    ow = offs_m % OW
    tmp = offs_m // OW
    oh = tmp % OH
    n_idx = tmp // OH

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # For each (oh, ow), figure out which (kh, kw, ih, iw) contribute.
    # Output: out[n, oc, oh, ow] = sum_{ic, kh, kw}
    #         input[n, ic, ih, iw] * weight[ic, oc, kh, kw]
    # where ih = (oh + PAD_H - kh) / STRIDE_H if divisible, similarly iw.
    for kh in tl.static_range(0, KH):
        ih_num = oh + PAD_H - kh
        ih = ih_num // STRIDE_H
        ih_valid = ((ih_num % STRIDE_H) == 0) & (ih >= 0) & (ih < IH)
        for kw in tl.static_range(0, KW):
            iw_num = ow + PAD_W - kw
            iw = iw_num // STRIDE_W
            iw_valid = ((iw_num % STRIDE_W) == 0) & (iw >= 0) & (iw < IW)
            valid = ih_valid & iw_valid & m_mask

            # Load input row block: shape [BLOCK_M, IC] via loop over IC tiles
            # Reduce over IC for this (kh, kw)
            # input offset base for each m: n_idx * IC * IH * IW + ih * IW + iw
            # weight offset base for each n (oc): oc * KH * KW + kh * KW + kw (weight layout [IC, OC, KH, KW])
            base_in = n_idx * (IC * IH * IW) + ih * IW + iw   # add ic * IH*IW
            base_w = offs_n * (KH * KW) + kh * KW + kw         # add ic * OC*KH*KW

            # Reduce over IC
            for ic in range(0, IC):
                x_off = base_in + ic * (IH * IW)
                w_off = base_w + ic * (OC * KH * KW)
                x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)  # [BLOCK_M]
                w_val = tl.load(w_ptr + w_off, mask=n_mask, other=0.0)  # [BLOCK_N]
                acc += x_val[:, None] * w_val[None, :]

    # Bias
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    # Mish: x * tanh(softplus(x))
    x = acc
    # stable softplus
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(tl.where(x > 20.0, 0.0, x))))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    y = x * th
    y = y + ADD_VAL
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)
    y = y * SCALE

    # Store: out[n, oc, oh, ow]
    out_off = n_idx[:, None] * (OC * OH * OW) + offs_n[None, :] * (OH * OW) + oh[:, None] * OW + ow[:, None]
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_off, y, mask=store_mask)


def conv_transpose_fused(x, weight, bias, stride, padding, output_padding, add_value, scale):
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    N, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w
    OH = (IH - 1) * stride - 2 * padding + KH + output_padding
    OW = (IW - 1) * stride - 2 * padding + KW + output_padding

    out = torch.empty((N, OC, OH, OW), dtype=x.dtype, device=x.device)

    M = N * OH * OW
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OC, meta['BLOCK_N']),
    )
    conv_transpose_fused_kernel[grid](
        x, weight, bias, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        stride, stride,
        padding, padding,
        float(add_value), float(scale),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        return conv_transpose_fused(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.add_value,
            self.scale,
        )