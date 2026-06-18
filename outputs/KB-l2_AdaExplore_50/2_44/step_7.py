import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # program ids: (n, oc, hw_tile)
    n = tl.program_id(0)
    oc = tl.program_id(1)
    hw_tile = tl.program_id(2)

    hw_offsets = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    oh = hw_offsets // OW
    ow = hw_offsets % OW
    valid_hw = hw_offsets < (OH * OW)

    acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)

    # For each output position (oh, ow), accumulate:
    # sum over ic, kh, kw of x[n, ic, ih, iw] * w[ic, oc, kh, kw]
    # where ih = (oh + PH - kh) / SH (must be integer, and in [0, IH))
    #       iw = (ow + PW - kw) / SW (must be integer, and in [0, IW))

    for kh in tl.static_range(0, KH):
        h_num = oh + PH - kh
        ih = h_num // SH
        h_valid = (h_num >= 0) & ((h_num % SH) == 0) & (ih >= 0) & (ih < IH)
        for kw in tl.static_range(0, KW):
            w_num = ow + PW - kw
            iw = w_num // SW
            w_valid = (w_num >= 0) & ((w_num % SW) == 0) & (iw >= 0) & (iw < IW)
            hw_valid = h_valid & w_valid & valid_hw

            # Loop over input channels in tiles
            for ic_start in range(0, IC, BLOCK_IC):
                ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                ic_mask = ic_offs < IC

                # x[n, ic, ih, iw]: shape (BLOCK_IC, BLOCK_HW)
                x_ptrs = x_ptr + n * (IC * IH * IW) + ic_offs[:, None] * (IH * IW) + ih[None, :] * IW + iw[None, :]
                x_mask = ic_mask[:, None] & hw_valid[None, :]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

                # w[ic, oc, kh, kw]: shape (BLOCK_IC,)
                w_ptrs = w_ptr + ic_offs * (OC * KH * KW) + oc * (KH * KW) + kh * KW + kw
                w_vals = tl.load(w_ptrs, mask=ic_mask, other=0.0)

                # Multiply and accumulate
                prod = x_vals * w_vals[:, None]
                acc += tl.sum(prod, axis=0)

    # Add bias
    bias = tl.load(b_ptr + oc)
    acc = acc + bias

    # Store
    out_ptrs = out_ptr + n * (OC * OH * OW) + oc * (OH * OW) + hw_offsets
    tl.store(out_ptrs, acc, mask=valid_hw)


def conv_transpose2d_triton(x, weight, bias, stride, padding, output_padding):
    N, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w

    SH, SW = stride, stride
    PH, PW = padding, padding
    OPH, OPW = output_padding, output_padding

    OH = (IH - 1) * SH - 2 * PH + KH + OPH
    OW = (IW - 1) * SW - 2 * PW + KW + OPW

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_HW = 128
    BLOCK_IC = 16
    grid = (N, OC, triton.cdiv(OH * OW, BLOCK_HW))

    conv_transpose_kernel[grid](
        x, weight, bias, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        SH, SW,
        PH, PW,
        BLOCK_IC=BLOCK_IC,
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
    )
    return out


@triton.jit
def mean_scale_kernel(
    inp_ptr, out_ptr,
    N, C, HW,
    scale,
    BLOCK: tl.constexpr,
):
    # one program per (n, c)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    base = n * (C * HW) + c * HW

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, HW, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < HW
        vals = tl.load(inp_ptr + base + offs, mask=mask, other=0.0)
        acc += vals

    total = tl.sum(acc, axis=0)
    result = total * scale / HW
    tl.store(out_ptr + pid, result)


def mean_scale_triton(x, scale):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty((N, C), device=x.device, dtype=x.dtype)
    grid = (N * C,)
    BLOCK = 1024
    mean_scale_kernel[grid](x, out, N, C, HW, scale, BLOCK=BLOCK, num_warps=4)
    return out.view(N, C, 1, 1)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.multiplier = multiplier
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous()
        weight = self.conv_transpose.weight.contiguous()
        bias = self.conv_transpose.bias.contiguous()
        y = conv_transpose2d_triton(x, weight, bias, self.stride, self.padding, self.output_padding)
        # Fused scale + mean over H,W
        out = mean_scale_triton(y, self.multiplier)
        return out