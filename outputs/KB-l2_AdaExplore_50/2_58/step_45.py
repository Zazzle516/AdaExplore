import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_fused_kernel(
    x_ptr,        # [N, IC, ID, IH, IW]
    w_ptr,        # [IC, OC, KD, KH, KW]
    cb_ptr,       # conv bias [OC]
    bias_ptr,     # scalar bias
    out_ptr,      # [N, 1, OD, OH, OW]
    N, IC: tl.constexpr, OC: tl.constexpr,
    ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_hw = tl.program_id(2)

    # Spatial tile within (OH, OW) for fixed (n, od)
    hw_start = pid_hw * BLOCK_SPATIAL
    hw_offs = hw_start + tl.arange(0, BLOCK_SPATIAL)  # [BLOCK]
    oh = hw_offs // OW
    ow = hw_offs % OW
    hw_mask = hw_offs < (OH * OW)

    od = pid_d

    # accumulator per OC: [BLOCK, OC]
    acc = tl.zeros((BLOCK_SPATIAL, OC), dtype=tl.float32)
    # Load conv bias [OC]
    oc_range = tl.arange(0, OC)
    cb = tl.load(cb_ptr + oc_range)  # [OC]
    acc = acc + cb[None, :]

    # Iterate over kernel positions
    # For ConvTranspose: out[od, oh, ow] = sum_{kd,kh,kw,ic} x[id, ih, iw] * w[ic, oc, kd, kh, kw]
    # where id*stride - pad + kd = od  =>  id = (od + pad - kd) / stride, valid if divisible and in range
    for kd in tl.static_range(0, KD):
        id_num = od + PAD - kd
        id_val = id_num // STRIDE
        id_valid = (id_num % STRIDE == 0) & (id_val >= 0) & (id_val < ID)
        for kh in tl.static_range(0, KH):
            ih_num = oh + PAD - kh
            ih_val = ih_num // STRIDE
            ih_valid = (ih_num % STRIDE == 0) & (ih_val >= 0) & (ih_val < IH)
            for kw in tl.static_range(0, KW):
                iw_num = ow + PAD - kw
                iw_val = iw_num // STRIDE
                iw_valid = (iw_num % STRIDE == 0) & (iw_val >= 0) & (iw_val < IW)

                spatial_valid = ih_valid & iw_valid & hw_mask  # [BLOCK]
                if id_valid:
                    # Load weight slice [IC, OC] for this (kd,kh,kw)
                    # w shape: [IC, OC, KD, KH, KW], stride: [OC*KD*KH*KW, KD*KH*KW, KH*KW, KW, 1]
                    ic_range = tl.arange(0, IC)
                    w_off = (ic_range[:, None] * (OC * KD * KH * KW)
                             + oc_range[None, :] * (KD * KH * KW)
                             + kd * (KH * KW) + kh * KW + kw)
                    w_vals = tl.load(w_ptr + w_off)  # [IC, OC]

                    # Load input [BLOCK, IC]
                    # x shape: [N, IC, ID, IH, IW]
                    x_base = (pid_n * (IC * ID * IH * IW)
                              + id_val * (IH * IW)
                              + ih_val * IW + iw_val)  # [BLOCK]
                    x_off = x_base[:, None] + ic_range[None, :] * (ID * IH * IW)
                    x_vals = tl.load(x_ptr + x_off, mask=spatial_valid[:, None], other=0.0)  # [BLOCK, IC]

                    # acc += x_vals @ w_vals
                    acc += tl.dot(x_vals, w_vals)

    # Now acc is [BLOCK, OC]: conv output for this output element across OC channels
    # LogSumExp over OC
    max_val = tl.max(acc, axis=1)  # [BLOCK]
    sum_exp = tl.sum(tl.exp(acc - max_val[:, None]), axis=1)
    lse = max_val + tl.log(sum_exp)

    # HardSwish: x * sigmoid(x+3) / 6
    hs = lse * tl.sigmoid(lse + 3.0) / 6.0

    # subtract bias
    b = tl.load(bias_ptr)
    out = hs - b

    # clamp
    out = tl.minimum(tl.maximum(out, -1.0), 1.0)

    # Store: output shape [N, 1, OD, OH, OW]
    out_off = (pid_n * (OD * OH * OW)
               + od * (OH * OW)
               + hw_offs)
    tl.store(out_ptr + out_off, out, mask=hw_mask)


def fused_convt3d_lse_hswish_sub_clamp(x, weight, conv_bias, bias,
                                       stride, padding, kernel_size):
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    OD = (ID - 1) * stride - 2 * padding + KD
    OH = (IH - 1) * stride - 2 * padding + KH
    OW = (IW - 1) * stride - 2 * padding + KW

    out = torch.empty((N, 1, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_SPATIAL = 64
    grid = (N, OD, (OH * OW + BLOCK_SPATIAL - 1) // BLOCK_SPATIAL)

    bias_flat = bias.contiguous().view(-1)[:1]
    cb_flat = conv_bias.contiguous()

    conv_transpose3d_fused_kernel[grid](
        x, weight, cb_flat, bias_flat, out,
        N, IC, OC, ID, IH, IW, OD, OH, OW,
        KD, KH, KW, stride, padding,
        BLOCK_SPATIAL=BLOCK_SPATIAL,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        return fused_convt3d_lse_hswish_sub_clamp(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.bias,
            self.stride,
            self.padding,
            self.kernel_size,
        )