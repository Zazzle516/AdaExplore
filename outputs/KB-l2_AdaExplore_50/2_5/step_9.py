import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Scatter-add style ConvTranspose2d:
# For each output tile (oh, ow) over a block of output channels, gather contributions
# from valid input positions and kernel taps. This is the inverse formulation of
# scatter and produces coalesced stores.
#
# For ConvTranspose2d with stride S, padding P, output_padding OP:
#   OH = (IH-1)*S - 2P + KH + OP
# At output position (oh, ow), input contribution comes from ih, iw where:
#   oh = ih*S - P + kh  =>  ih = (oh + P - kh) / S  (must be integer)
#   iw = (ow + P - kw) / S
# Need 0 <= ih < IH, 0 <= iw < IW

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 32, 'BLOCK_OC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 32, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 16, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 32, 'BLOCK_OC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 64, 'BLOCK_OC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 64, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 32, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 32, 'BLOCK_OW': 32, 'BLOCK_OC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 32, 'BLOCK_OC': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 16, 'BLOCK_OC': 32}, num_warps=2, num_stages=2),
    ],
    key=['IC', 'OC', 'IH', 'IW', 'KH', 'KW', 'STRIDE', 'PAD'],
)
@triton.jit
def conv_transpose2d_fused_kernel(
    inp_ptr,    # [N, IC, IH, IW]
    weight_ptr, # [IC, OC, KH, KW]
    bias_ptr,   # [OC]  effective bias = extra_bias - conv_bias
    out_ptr,    # [N, OC, OH, OW]
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_spatial = tl.program_id(1)
    pid_oc = tl.program_id(2)

    num_ow_blocks = tl.cdiv(OW, BLOCK_OW)
    num_oh_blocks = tl.cdiv(OH, BLOCK_OH)
    pid_oh = pid_spatial // num_ow_blocks
    pid_ow = pid_spatial % num_ow_blocks

    oh_offs = pid_oh * BLOCK_OH + tl.arange(0, BLOCK_OH)  # [BLOCK_OH]
    ow_offs = pid_ow * BLOCK_OW + tl.arange(0, BLOCK_OW)  # [BLOCK_OW]
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]

    oh_mask = oh_offs < OH
    ow_mask = ow_offs < OW
    oc_mask = oc_offs < OC

    # accumulator: [BLOCK_OH, BLOCK_OW, BLOCK_OC] - flatten spatial to 2D dot
    # Use shape [BLOCK_OH * BLOCK_OW, BLOCK_OC]
    M = BLOCK_OH * BLOCK_OW
    acc = tl.zeros((M, BLOCK_OC), dtype=tl.float32)

    # Flattened spatial offsets
    # oh_idx[m] = oh_offs[m // BLOCK_OW], ow_idx[m] = ow_offs[m % BLOCK_OW]
    m_range = tl.arange(0, M)
    oh_local = m_range // BLOCK_OW
    ow_local = m_range % BLOCK_OW
    # Gather oh_idx and ow_idx via arithmetic
    oh_idx = pid_oh * BLOCK_OH + oh_local
    ow_idx = pid_ow * BLOCK_OW + ow_local
    spatial_mask = (oh_idx < OH) & (ow_idx < OW)

    inp_n_base = pid_n * IC * IH * IW

    # Loop over kernel taps
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih_num = oh_idx + PAD - kh
            iw_num = ow_idx + PAD - kw
            ih = ih_num // STRIDE
            iw = iw_num // STRIDE
            valid_h = (ih_num >= 0) & ((ih * STRIDE) == ih_num) & (ih < IH)
            valid_w = (iw_num >= 0) & ((iw * STRIDE) == iw_num) & (iw < IW)
            valid = valid_h & valid_w & spatial_mask  # [M]

            # Loop over IC in blocks
            BLOCK_K: tl.constexpr = 32
            for k0 in range(0, IC, BLOCK_K):
                k_offs = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
                k_mask = k_offs < IC

                # input addresses: [M, BLOCK_K]
                # offset = n*IC*IH*IW + k*IH*IW + ih*IW + iw
                inp_addr = (inp_n_base
                            + k_offs[None, :] * (IH * IW)
                            + ih[:, None] * IW
                            + iw[:, None])
                a_mask = valid[:, None] & k_mask[None, :]
                a = tl.load(inp_ptr + inp_addr, mask=a_mask, other=0.0)

                # weight addresses: [BLOCK_K, BLOCK_OC]
                # offset = k*OC*KH*KW + oc*KH*KW + kh*KW + kw
                w_addr = (k_offs[:, None] * (OC * KH * KW)
                          + oc_offs[None, :] * (KH * KW)
                          + kh * KW + kw)
                w_mask = k_mask[:, None] & oc_mask[None, :]
                w = tl.load(weight_ptr + w_addr, mask=w_mask, other=0.0)

                acc += tl.dot(a, w)

    # Epilogue: subtract effective bias, apply tanh
    b = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc = acc - b[None, :]
    # tanh via exp(2x)
    two_acc = 2.0 * acc
    # clamp to avoid overflow
    two_acc = tl.minimum(tl.maximum(two_acc, -40.0), 40.0)
    e2 = tl.exp(two_acc)
    acc = (e2 - 1.0) / (e2 + 1.0)

    # Store output[n, oc, oh, ow]
    out_n_base = pid_n * OC * OH * OW
    out_addr = (out_n_base
                + oc_offs[None, :] * (OH * OW)
                + oh_idx[:, None] * OW
                + ow_idx[:, None])
    store_mask = spatial_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_addr, acc, mask=store_mask)


def conv_transpose2d_fused(x, weight, conv_bias, extra_bias, stride, padding, output_padding):
    N, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w

    OH = (IH - 1) * stride - 2 * padding + KH + output_padding
    OW = (IW - 1) * stride - 2 * padding + KW + output_padding

    if conv_bias is not None:
        eff_bias = (extra_bias - conv_bias).contiguous()
    else:
        eff_bias = extra_bias.contiguous()

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        N,
        triton.cdiv(OH, META['BLOCK_OH']) * triton.cdiv(OW, META['BLOCK_OW']),
        triton.cdiv(OC, META['BLOCK_OC']),
    )

    conv_transpose2d_fused_kernel[grid](
        x, weight, eff_bias, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        stride, padding,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()
        cb = self.conv_transpose.bias
        eb = self.bias.view(-1).contiguous()
        return conv_transpose2d_fused(
            x, w, cb, eb,
            self.stride, self.padding, self.output_padding,
        )