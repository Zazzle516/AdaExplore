import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_fused_kernel(
    x_ptr, w_ptr, conv_bias_ptr, extra_bias_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_S: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)

    OHW = OH * OW
    OS = OD * OHW

    # decode output spatial coords
    od = offs_s // OHW
    rem = offs_s - od * OHW
    oh = rem // OW
    ow = rem - oh * OW

    s_mask = offs_s < OS
    oc_mask = offs_oc < OC

    acc = tl.zeros((BLOCK_OC, BLOCK_S), dtype=tl.float32)

    # For each kernel position
    for kd in tl.static_range(KD):
        for kh in tl.static_range(KH):
            for kw in tl.static_range(KW):
                # id*stride = od + pad - kd  =>  id = (od + pad - kd)/stride
                num_d = od + PAD - kd
                num_h = oh + PAD - kh
                num_w = ow + PAD - kw

                id_ = num_d // STRIDE
                ih_ = num_h // STRIDE
                iw_ = num_w // STRIDE

                valid_d = (num_d % STRIDE == 0) & (id_ >= 0) & (id_ < ID)
                valid_h = (num_h % STRIDE == 0) & (ih_ >= 0) & (ih_ < IH)
                valid_w = (num_w % STRIDE == 0) & (iw_ >= 0) & (iw_ < IW)
                valid = valid_d & valid_h & valid_w & s_mask  # [BLOCK_S]

                # input offset base for this (n, ic=?, id, ih, iw)
                # We'll loop over IC in chunks and do matmul-like accumulation
                # input[n, ic, id, ih, iw] -> ptr = n*IC*ID*IH*IW + ic*ID*IH*IW + id*IH*IW + ih*IW + iw
                # weight[ic, oc, kd, kh, kw] -> ptr = ic*OC*KD*KH*KW + oc*KD*KH*KW + kd*KH*KW + kh*KW + kw
                in_spatial = id_ * IH * IW + ih_ * IW + iw_  # [BLOCK_S]
                w_kpos = kd * KH * KW + kh * KW + kw

                for ic_start in range(0, IC, BLOCK_IC):
                    offs_ic = ic_start + tl.arange(0, BLOCK_IC)
                    ic_mask = offs_ic < IC

                    # load input [BLOCK_IC, BLOCK_S]
                    in_ptrs = (x_ptr
                               + pid_n * IC * ID * IH * IW
                               + offs_ic[:, None] * (ID * IH * IW)
                               + in_spatial[None, :])
                    in_vals = tl.load(in_ptrs,
                                      mask=ic_mask[:, None] & valid[None, :],
                                      other=0.0)

                    # load weight [BLOCK_IC, BLOCK_OC]
                    w_ptrs = (w_ptr
                              + offs_ic[:, None] * (OC * KD * KH * KW)
                              + offs_oc[None, :] * (KD * KH * KW)
                              + w_kpos)
                    w_vals = tl.load(w_ptrs,
                                     mask=ic_mask[:, None] & oc_mask[None, :],
                                     other=0.0)

                    # acc[BLOCK_OC, BLOCK_S] += w_vals.T @ in_vals
                    acc += tl.dot(tl.trans(w_vals), in_vals)

    # add conv bias
    cb = tl.load(conv_bias_ptr + offs_oc, mask=oc_mask, other=0.0)
    eb = tl.load(extra_bias_ptr + offs_oc, mask=oc_mask, other=0.0)
    x = acc + cb[:, None]
    bias_total = eb[:, None]
    # epilogue: (2x + bias)*x + x
    res = (2.0 * x + bias_total) * x + x

    # store
    out_base = (pid_n * OC * OS
                + offs_oc[:, None] * OS
                + offs_s[None, :])
    out_mask = oc_mask[:, None] & s_mask[None, :]
    tl.store(out_ptr + out_base, res, mask=out_mask)


def conv_transpose3d_fused(x, weight, conv_bias, extra_bias,
                           stride, padding, output_padding):
    x = x.contiguous()
    weight = weight.contiguous()
    N, IC, ID, IH, IW = x.shape
    IC2, OC, KD, KH, KW = weight.shape
    assert IC == IC2

    OD = (ID - 1) * stride - 2 * padding + KD + output_padding
    OH = (IH - 1) * stride - 2 * padding + KH + output_padding
    OW = (IW - 1) * stride - 2 * padding + KW + output_padding

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    OS = OD * OH * OW
    BLOCK_OC = 32
    BLOCK_S = 64
    BLOCK_IC = 32

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OS, BLOCK_S))

    conv_transpose3d_fused_kernel[grid](
        x, weight, conv_bias, extra_bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        stride, padding,
        BLOCK_OC, BLOCK_S, BLOCK_IC,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        return conv_transpose3d_fused(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.bias.view(-1),
            self.stride, self.padding, self.output_padding,
        )