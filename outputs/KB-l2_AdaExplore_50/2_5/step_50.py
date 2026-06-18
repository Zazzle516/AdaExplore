import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _scatter_convtranspose_kernel(
    x_ptr,        # [N, IC, IH, IW]
    w_ptr,        # [IC, OC, KH, KW]
    out_ptr,      # [N, OC, OH, OW] - pre-initialized with conv_bias - sub_bias
    N, IC, OC,
    IH, IW, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, ih, iw, ic)
    pid = tl.program_id(0)
    iw = pid % IW
    pid = pid // IW
    ih = pid % IH
    pid = pid // IH
    ic = pid % IC
    n = pid // IC

    # load input scalar
    x_off = ((n * IC + ic) * IH + ih) * IW + iw
    x_val = tl.load(x_ptr + x_off)

    offs_oc = tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    # for each kernel position, scatter add x_val * w[ic, :, kh, kw] into out
    for kh in tl.static_range(0, KH):
        oh = ih * STRIDE - PAD + kh
        for kw in tl.static_range(0, KW):
            ow = iw * STRIDE - PAD + kw
            in_bounds = (oh >= 0) & (oh < OH) & (ow >= 0) & (ow < OW)
            # weight [IC, OC, KH, KW]
            w_off = ((ic * OC + offs_oc) * KH + kh) * KW + kw
            w_vals = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)
            contrib = x_val * w_vals
            out_off = ((n * OC + offs_oc) * OH + oh) * OW + ow
            store_mask = mask_oc & in_bounds
            tl.atomic_add(out_ptr + out_off, contrib, mask=store_mask)


@triton.jit
def _tanh_inplace_kernel(out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(out_ptr + offs, mask=mask, other=0.0)
    # tanh via exp
    e2x = tl.exp(2.0 * x)
    y = (e2x - 1.0) / (e2x + 1.0)
    tl.store(out_ptr + offs, y, mask=mask)


def conv_transpose2d_triton(x, weight, conv_bias, sub_bias, stride, padding, output_padding):
    N, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w

    OH = (IH - 1) * stride - 2 * padding + KH + output_padding
    OW = (IW - 1) * stride - 2 * padding + KW + output_padding

    # initial value: conv_bias (per OC) - sub_bias (broadcast over OC,1,1)
    # sub_bias has shape (OC, 1, 1)
    init_bias = conv_bias.view(1, OC, 1, 1) - sub_bias.view(1, OC, 1, 1)
    out = init_bias.expand(N, OC, OH, OW).contiguous()

    BLOCK_OC = triton.next_power_of_2(OC)
    if BLOCK_OC < 16:
        BLOCK_OC = 16

    grid = (N * IC * IH * IW,)
    _scatter_convtranspose_kernel[grid](
        x, weight, out,
        N, IC, OC,
        IH, IW, OH, OW,
        KH, KW,
        stride, padding,
        BLOCK_OC=BLOCK_OC,
        num_warps=4,
    )

    n_elements = out.numel()
    BLOCK = 1024
    grid2 = ((n_elements + BLOCK - 1) // BLOCK,)
    _tanh_inplace_kernel[grid2](out, n_elements, BLOCK=BLOCK, num_warps=4)
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
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous()
        conv_bias = self.conv_transpose.bias.contiguous()
        sub_bias = self.bias.contiguous()
        return conv_transpose2d_triton(
            x, weight, conv_bias, sub_bias,
            self.stride, self.padding, self.output_padding
        )