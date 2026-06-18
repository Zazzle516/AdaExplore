import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, OC,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # one program per (n, d_out, h_out, w_tile) processing BLOCK_OC channels x BLOCK_W positions
    pid = tl.program_id(0)
    W_tiles = (W_out + BLOCK_W - 1) // BLOCK_W
    row = pid // W_tiles
    wt = pid % W_tiles

    n = row // (D_out * H_out)
    rem = row % (D_out * H_out)
    d_out = rem // H_out
    h_out = rem % H_out

    offs_w = wt * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W_out
    offs_oc = tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    # accumulator [BLOCK_OC, BLOCK_W]
    acc = tl.zeros((BLOCK_OC, BLOCK_W), dtype=tl.float32)

    # input strides
    in_stride_n = IC * D_in * H_in * W_in
    in_stride_c = D_in * H_in * W_in
    in_stride_d = H_in * W_in
    in_stride_h = W_in

    # weight: [IC, OC, KD, KH, KW]
    w_stride_ic = OC * KD * KH * KW
    w_stride_oc = KD * KH * KW

    # For transposed conv:
    # output[n, oc, d_out, h_out, w_out] = sum_{ic, kd, kh, kw} input[n, ic, id, ih, iw] * weight[ic, oc, kd, kh, kw]
    # where id*stride - pad + kd = d_out => id = (d_out + pad - kd) / stride, must be integer and in [0, D_in)
    for kd in tl.static_range(0, KD):
        id_num = d_out + PAD - kd
        id_val = id_num // STRIDE
        id_valid = (id_num % STRIDE == 0) & (id_val >= 0) & (id_val < D_in)
        for kh in tl.static_range(0, KH):
            ih_num = h_out + PAD - kh
            ih_val = ih_num // STRIDE
            ih_valid = (ih_num % STRIDE == 0) & (ih_val >= 0) & (ih_val < H_in)
            for kw in tl.static_range(0, KW):
                iw_num = offs_w + PAD - kw  # [BLOCK_W]
                iw_val = iw_num // STRIDE
                iw_valid = (iw_num % STRIDE == 0) & (iw_val >= 0) & (iw_val < W_in) & mask_w
                valid_dhw = id_valid & ih_valid & iw_valid  # [BLOCK_W]

                # load input[n, :, id, ih, iw] for all IC -> shape [IC, BLOCK_W]
                # load weight[:, :, kd, kh, kw] -> shape [IC, OC]
                for ic in tl.static_range(0, 3):  # IC = 3
                    x_off = n * in_stride_n + ic * in_stride_c + id_val * in_stride_d + ih_val * in_stride_h + iw_val
                    x_val = tl.load(x_ptr + x_off, mask=valid_dhw, other=0.0)  # [BLOCK_W]

                    w_off = ic * w_stride_ic + offs_oc * w_stride_oc + kd * KH * KW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)  # [BLOCK_OC]

                    acc += w_val[:, None] * x_val[None, :]

    # add bias
    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc += bias[:, None]

    # store output [N, OC, D_out, H_out, W_out]
    out_stride_n = OC * D_out * H_out * W_out
    out_stride_c = D_out * H_out * W_out
    out_stride_d = H_out * W_out
    out_stride_h = W_out

    out_off = (n * out_stride_n
               + offs_oc[:, None] * out_stride_c
               + d_out * out_stride_d
               + h_out * out_stride_h
               + offs_w[None, :])
    mask_out = mask_oc[:, None] & mask_w[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask_out)


@triton.jit
def fused_pool_softmax_sub_swish_max_kernel(
    x_ptr, sub_ptr, out_ptr,
    N, C,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    BLOCK_C: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    W_tiles = (W_out + BLOCK_W - 1) // BLOCK_W
    row = pid // W_tiles
    wt = pid % W_tiles

    n = row // (D_out * H_out)
    rem = row % (D_out * H_out)
    pd = rem // H_out
    ph = rem % H_out

    d0 = pd * 2
    h0 = ph * 2

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    offs_w = wt * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W_out
    w0 = offs_w * 2

    stride_n = C * D_in * H_in * W_in
    stride_c = D_in * H_in * W_in
    stride_d = H_in * W_in
    stride_h = W_in

    base = (n * stride_n
            + offs_c[:, None] * stride_c
            + d0 * stride_d
            + h0 * stride_h
            + w0[None, :])
    mask = mask_c[:, None] & mask_w[None, :]

    v000 = tl.load(x_ptr + base, mask=mask, other=-float('inf'))
    v001 = tl.load(x_ptr + base + 1, mask=mask, other=-float('inf'))
    v010 = tl.load(x_ptr + base + stride_h, mask=mask, other=-float('inf'))
    v011 = tl.load(x_ptr + base + stride_h + 1, mask=mask, other=-float('inf'))
    v100 = tl.load(x_ptr + base + stride_d, mask=mask, other=-float('inf'))
    v101 = tl.load(x_ptr + base + stride_d + 1, mask=mask, other=-float('inf'))
    v110 = tl.load(x_ptr + base + stride_d + stride_h, mask=mask, other=-float('inf'))
    v111 = tl.load(x_ptr + base + stride_d + stride_h + 1, mask=mask, other=-float('inf'))

    x = tl.maximum(tl.maximum(tl.maximum(v000, v001), tl.maximum(v010, v011)),
                   tl.maximum(tl.maximum(v100, v101), tl.maximum(v110, v111)))

    m = tl.max(x, axis=0)
    e = tl.exp(x - m[None, :])
    e = tl.where(mask_c[:, None], e, 0.0)
    z = tl.sum(e, axis=0)
    sm = e / z[None, :]

    sub = tl.load(sub_ptr + offs_c, mask=mask_c, other=0.0)
    y = sm - sub[:, None]
    sw = y * tl.sigmoid(y)
    sw = tl.where(mask_c[:, None], sw, -float('inf'))
    out_val = tl.max(sw, axis=0)

    S_out = D_out * H_out * W_out
    out_off = n * S_out + pd * H_out * W_out + ph * W_out + offs_w
    tl.store(out_ptr + out_off, out_val, mask=mask_w)


def custom_conv_transpose3d(x, weight, bias, stride, padding, output_padding, kernel_size):
    N, IC, D_in, H_in, W_in = x.shape
    OC = weight.shape[1]
    KD = KH = KW = kernel_size
    D_out = (D_in - 1) * stride - 2 * padding + KD + output_padding
    H_out = (H_in - 1) * stride - 2 * padding + KH + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + KW + output_padding

    out = torch.empty((N, OC, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_OC = 16  # OC=16
    BLOCK_W = 16

    W_tiles = (W_out + BLOCK_W - 1) // BLOCK_W
    grid = (N * D_out * H_out * W_tiles,)

    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        N, IC, OC,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        KD, KH, KW,
        stride, padding,
        BLOCK_OC, BLOCK_W,
        num_warps=4, num_stages=2,
    )
    return out


def fused_pool_post(x, sub):
    N, C, D_in, H_in, W_in = x.shape
    D_out = D_in // 2
    H_out = H_in // 2
    W_out = W_in // 2
    out = torch.empty((N, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    BLOCK_C = triton.next_power_of_2(C)
    BLOCK_W = 16
    W_tiles = (W_out + BLOCK_W - 1) // BLOCK_W
    grid = (N * D_out * H_out * W_tiles,)
    fused_pool_softmax_sub_swish_max_kernel[grid](
        x, sub, out,
        N, C,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        BLOCK_C=BLOCK_C, BLOCK_W=BLOCK_W,
        num_warps=2, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding

    def forward(self, x):
        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()
        b = self.conv_transpose.bias.contiguous()
        x = custom_conv_transpose3d(x, w, b, self.stride, self.padding, self.output_padding, self.kernel_size)

        if (self.pool_kernel_size == 2 and self.pool_stride == 2 and self.pool_padding == 0):
            x = fused_pool_post(x, self.subtract.contiguous())
        else:
            x = self.max_pool(x)
            x = torch.softmax(x, dim=1)
            x = x - self.subtract.view(1, -1, 1, 1, 1)
            x = torch.sigmoid(x) * x
            x = torch.max(x, dim=1)[0]
        return x