import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_fused_kernel(
    x_ptr, w_ptr, b_ptr, add_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_SP: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    OHW = OH * OW
    ODHW = OD * OH * OW

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < ODHW

    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # accumulator [BLOCK_OC, BLOCK_SP]
    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # K = IC * KD * KH * KW
    K_total = IC * KD * KH * KW

    # iterate over K in chunks
    for k_start in range(0, K_total, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K_total

        ic = k_offs // (KD * KH * KW)
        krem = k_offs % (KD * KH * KW)
        kd = krem // (KH * KW)
        krem2 = krem % (KH * KW)
        kh = krem2 // KW
        kw = krem2 % KW

        # weight: [IC, OC, KD, KH, KW]
        # load weight[ic, oc, kd, kh, kw] -> [BLOCK_K, BLOCK_OC]
        w_offs = (ic[:, None] * OC * KD * KH * KW
                  + oc_offs[None, :] * KD * KH * KW
                  + kd[:, None] * KH * KW
                  + kh[:, None] * KW
                  + kw[:, None])
        w_mask = k_mask[:, None] & oc_mask[None, :]
        w = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_OC]

        # for each (output spatial, k), compute corresponding input position
        # od + PAD - kd = id_num, must be divisible by STRIDE and in [0, ID)
        id_num = od[None, :] + PAD - kd[:, None]   # [BLOCK_K, BLOCK_SP]
        ih_num = oh[None, :] + PAD - kh[:, None]
        iw_num = ow[None, :] + PAD - kw[:, None]

        id_div = id_num // STRIDE
        ih_div = ih_num // STRIDE
        iw_div = iw_num // STRIDE

        valid = ((id_num % STRIDE) == 0) & ((ih_num % STRIDE) == 0) & ((iw_num % STRIDE) == 0)
        valid = valid & (id_div >= 0) & (id_div < ID)
        valid = valid & (ih_div >= 0) & (ih_div < IH)
        valid = valid & (iw_div >= 0) & (iw_div < IW)
        valid = valid & k_mask[:, None] & sp_mask[None, :]

        x_offs = (pid_n * IC * ID * IH * IW
                  + ic[:, None] * ID * IH * IW
                  + id_div * IH * IW
                  + ih_div * IW
                  + iw_div)

        x_vals = tl.load(x_ptr + x_offs, mask=valid, other=0.0)  # [BLOCK_K, BLOCK_SP]

        # accumulate: acc[oc, sp] += sum_k w[k, oc] * x[k, sp]
        # use tl.dot: w.T @ x => [BLOCK_OC, BLOCK_SP]
        acc += tl.dot(tl.trans(w), x_vals)

    # add bias and add_input, then v * hardswish(v)
    # bias [OC]
    bias_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc = acc + bias_vals[:, None]

    # add_input: [N, OC, OD, OH, OW]
    add_offs = (pid_n * OC * ODHW
                + oc_offs[:, None] * ODHW
                + sp_offs[None, :])
    add_mask = oc_mask[:, None] & sp_mask[None, :]
    add_vals = tl.load(add_ptr + add_offs, mask=add_mask, other=0.0)
    v = acc + add_vals

    relu6 = tl.minimum(tl.maximum(v + 3.0, 0.0), 6.0)
    hs = v * relu6 * (1.0 / 6.0)
    out = v * hs

    tl.store(out_ptr + add_offs, out, mask=add_mask)


def conv_transpose3d_fused(x, weight, bias, add_input, stride, padding, output_padding):
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    OD = (ID - 1) * stride - 2 * padding + KD + output_padding
    OH = (IH - 1) * stride - 2 * padding + KH + output_padding
    OW = (IW - 1) * stride - 2 * padding + KW + output_padding

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    add_input = add_input.contiguous()
    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 64
    BLOCK_SP = 64
    BLOCK_K = 32

    ODHW = OD * OH * OW
    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(ODHW, BLOCK_SP))

    conv_transpose3d_fused_kernel[grid](
        x, weight, bias, add_input, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        stride, padding,
        BLOCK_OC, BLOCK_SP, BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding,
                                                  output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x, add_input):
        weight = self.conv_transpose.weight
        conv_bias = self.conv_transpose.bias
        return conv_transpose3d_fused(
            x, weight, conv_bias, add_input,
            self.stride, self.padding, self.output_padding,
        )