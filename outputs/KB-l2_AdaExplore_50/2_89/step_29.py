import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 8}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_W': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_W': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 16}, num_warps=8, num_stages=2),
    ],
    key=['N', 'C', 'D_in', 'H_in', 'W_in'],
)
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


def fused_pool_post(x, sub):
    N, C, D_in, H_in, W_in = x.shape
    D_out = D_in // 2
    H_out = H_in // 2
    W_out = W_in // 2
    x_c = x.contiguous()
    out = torch.empty((N, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    BLOCK_C = triton.next_power_of_2(C)
    grid = lambda META: (N * D_out * H_out * ((W_out + META['BLOCK_W'] - 1) // META['BLOCK_W']),)
    fused_pool_softmax_sub_swish_max_kernel[grid](
        x_c, sub.contiguous(), out,
        N, C,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        BLOCK_C=BLOCK_C,
    )
    return out


# ConvTranspose3d as gather GEMM:
# output[n, oc, od, oh, ow] = bias[oc] + sum_{ic, kd, kh, kw} weight[ic, oc, kd, kh, kw] * input[n, ic, id, ih, iw]
# where id = (od + pad - kd) / stride, only if divisible and in bounds.
# Weight layout for ConvTranspose3d: (in_channels, out_channels, kD, kH, kW)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SP': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 256}, num_warps=8, num_stages=2),
    ],
    key=['N', 'IC', 'OC', 'D_out', 'H_out', 'W_out', 'KD', 'KH', 'KW'],
)
@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, OC,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    stride_s: tl.constexpr,
    pad: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    # grid: (ceil(S_out/BLOCK_SP), N)
    pid_sp = tl.program_id(0)
    n = tl.program_id(1)

    S_out = D_out * H_out * W_out
    offs_sp = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]
    mask_sp = offs_sp < S_out

    # Decompose linear sp into (od, oh, ow)
    od = offs_sp // (H_out * W_out)
    rem = offs_sp % (H_out * W_out)
    oh = rem // W_out
    ow = rem % W_out

    offs_oc = tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    mask_oc = offs_oc < OC

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    x_stride_n = IC * D_in * H_in * W_in
    x_stride_c = D_in * H_in * W_in
    x_stride_d = H_in * W_in
    x_stride_h = W_in

    # weight: (IC, OC, KD, KH, KW)
    w_stride_ic = OC * KD * KH * KW
    w_stride_oc = KD * KH * KW
    w_stride_kd = KH * KW
    w_stride_kh = KW

    for kd in tl.static_range(0, KD):
        id_num = od + pad - kd  # [BLOCK_SP]
        id_v = id_num // stride_s
        valid_d = (id_num % stride_s == 0) & (id_v >= 0) & (id_v < D_in)
        for kh in tl.static_range(0, KH):
            ih_num = oh + pad - kh
            ih_v = ih_num // stride_s
            valid_h = (ih_num % stride_s == 0) & (ih_v >= 0) & (ih_v < H_in)
            for kw in tl.static_range(0, KW):
                iw_num = ow + pad - kw
                iw_v = iw_num // stride_s
                valid_w = (iw_num % stride_s == 0) & (iw_v >= 0) & (iw_v < W_in)
                valid = valid_d & valid_h & valid_w & mask_sp  # [BLOCK_SP]

                # x base for this spatial: n*x_stride_n + ic*x_stride_c + id*x_stride_d + ih*x_stride_h + iw
                x_sp_off = id_v * x_stride_d + ih_v * x_stride_h + iw_v  # [BLOCK_SP]

                for ic in range(0, IC):
                    x_off = n * x_stride_n + ic * x_stride_c + x_sp_off  # [BLOCK_SP]
                    x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)  # [BLOCK_SP]

                    w_off = ic * w_stride_ic + offs_oc * w_stride_oc + kd * w_stride_kd + kh * w_stride_kh + kw  # [BLOCK_OC]
                    w_val = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)  # [BLOCK_OC]

                    acc += w_val[:, None] * x_val[None, :]

    # add bias
    b = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc += b[:, None]

    # store: out[n, oc, od, oh, ow]
    out_stride_n = OC * S_out
    out_stride_c = S_out
    out_off = n * out_stride_n + offs_oc[:, None] * out_stride_c + offs_sp[None, :]
    mask_out = mask_oc[:, None] & mask_sp[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask_out)


def conv_transpose3d_triton(x, weight, bias, stride, padding, output_padding):
    N, IC, D_in, H_in, W_in = x.shape
    IC2, OC, KD, KH, KW = weight.shape
    assert IC == IC2
    D_out = (D_in - 1) * stride - 2 * padding + KD + output_padding
    H_out = (H_in - 1) * stride - 2 * padding + KH + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + KW + output_padding

    x_c = x.contiguous()
    w_c = weight.contiguous()
    b_c = bias.contiguous() if bias is not None else torch.zeros(OC, device=x.device, dtype=x.dtype)

    out = torch.empty((N, OC, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    BLOCK_OC = triton.next_power_of_2(OC)
    S_out = D_out * H_out * W_out
    grid = lambda META: ((S_out + META['BLOCK_SP'] - 1) // META['BLOCK_SP'], N)

    conv_transpose3d_kernel[grid](
        x_c, w_c, b_c, out,
        N, IC, OC,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        KD=KD, KH=KH, KW=KW,
        stride_s=stride,
        pad=padding,
        BLOCK_OC=BLOCK_OC,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding

    def forward(self, x):
        x = x.contiguous()
        x = conv_transpose3d_triton(
            x, self.conv_transpose.weight, self.conv_transpose.bias,
            self.stride, self.padding, self.output_padding
        )
        if (self.pool_kernel_size == 2 and self.pool_stride == 2 and self.pool_padding == 0):
            x = fused_pool_post(x, self.subtract)
        else:
            x = self.max_pool(x)
            x = torch.softmax(x, dim=1)
            x = x - self.subtract.view(1, -1, 1, 1, 1)
            x = torch.sigmoid(x) * x
            x = torch.max(x, dim=1)[0]
        return x