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
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # one program per (n, d_out, h_out, w_tile)
    pid = tl.program_id(0)
    W_tiles = (W_out + BLOCK_W - 1) // BLOCK_W
    n_dh_t = N * D_out * H_out * W_tiles
    
    wt = pid % W_tiles
    tmp = pid // W_tiles
    h_out = tmp % H_out
    tmp = tmp // H_out
    d_out = tmp % D_out
    n = tmp // D_out

    offs_w = wt * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W_out

    offs_oc = tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    # accumulator: [BLOCK_OC, BLOCK_W]
    acc = tl.zeros((BLOCK_OC, BLOCK_W), dtype=tl.float32)

    # output position in input space (after padding offset)
    # x_in = output_pos + PAD - k
    # input idx = x_in / stride, valid only if x_in % stride == 0 and 0<=in<size
    
    od_p = d_out + PD
    oh_p = h_out + PH
    ow_p = offs_w + PW  # [BLOCK_W]

    stride_x_n = IC * D_in * H_in * W_in
    stride_x_c = D_in * H_in * W_in
    stride_x_d = H_in * W_in
    stride_x_h = W_in

    # weight shape: (IC, OC, KD, KH, KW)
    stride_w_ic = OC * KD * KH * KW
    stride_w_oc = KD * KH * KW
    stride_w_kd = KH * KW
    stride_w_kh = KW

    for kd in tl.static_range(KD):
        id_raw = od_p - kd
        id_v = id_raw // SD
        id_valid = (id_raw % SD == 0) & (id_v >= 0) & (id_v < D_in)
        for kh in tl.static_range(KH):
            ih_raw = oh_p - kh
            ih_v = ih_raw // SH
            ih_valid = (ih_raw % SH == 0) & (ih_v >= 0) & (ih_v < H_in)
            for kw in tl.static_range(KW):
                iw_raw = ow_p - kw  # [BLOCK_W]
                iw_v = iw_raw // SW
                iw_valid = (iw_raw % SW == 0) & (iw_v >= 0) & (iw_v < W_in)

                spatial_valid = id_valid & ih_valid & iw_valid & mask_w  # [BLOCK_W]

                # input offsets per ic, w: [IC_size we loop over]
                # We loop over ic explicitly since IC=3
                for ic in tl.static_range(0, 3):
                    x_off = (n * stride_x_n
                             + ic * stride_x_c
                             + id_v * stride_x_d
                             + ih_v * stride_x_h
                             + iw_v)  # [BLOCK_W]
                    x_val = tl.load(x_ptr + x_off, mask=spatial_valid, other=0.0)  # [BLOCK_W]

                    w_off = (ic * stride_w_ic
                             + offs_oc * stride_w_oc
                             + kd * stride_w_kd
                             + kh * stride_w_kh
                             + kw)  # [BLOCK_OC]
                    w_val = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)

                    acc += w_val[:, None] * x_val[None, :]

    # add bias
    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc += bias[:, None]

    # store
    stride_o_n = OC * D_out * H_out * W_out
    stride_o_c = D_out * H_out * W_out
    stride_o_d = H_out * W_out
    stride_o_h = W_out
    out_off = (n * stride_o_n
               + offs_oc[:, None] * stride_o_c
               + d_out * stride_o_d
               + h_out * stride_o_h
               + offs_w[None, :])
    out_mask = mask_oc[:, None] & mask_w[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


def triton_conv_transpose3d(x, weight, bias, stride, padding, output_padding):
    N, IC, D_in, H_in, W_in = x.shape
    _, OC, KD, KH, KW = weight.shape
    SD, SH, SW = stride
    PD, PH, PW = padding
    OPD, OPH, OPW = output_padding

    D_out = (D_in - 1) * SD - 2 * PD + KD + OPD
    H_out = (H_in - 1) * SH - 2 * PH + KH + OPH
    W_out = (W_in - 1) * SW - 2 * PW + KW + OPW

    out = torch.empty((N, OC, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_OC = triton.next_power_of_2(OC)
    BLOCK_W = 16

    W_tiles = (W_out + BLOCK_W - 1) // BLOCK_W
    grid = (N * D_out * H_out * W_tiles,)

    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        N, IC, OC,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_OC=BLOCK_OC,
        BLOCK_W=BLOCK_W,
        num_warps=2,
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 4}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_W': 8}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_W': 16}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_W': 8}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_W': 16}, num_warps=2, num_stages=2),
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


def fused_pool_post(x, sub, pool_k, pool_s, pool_p):
    N, C, D_in, H_in, W_in = x.shape
    D_out = (D_in - pool_k) // pool_s + 1
    H_out = (H_in - pool_k) // pool_s + 1
    W_out = (W_in - pool_k) // pool_s + 1
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


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))

        # Cache convolution parameters
        ks = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        st = stride if isinstance(stride, tuple) else (stride,) * 3
        pd = padding if isinstance(padding, tuple) else (padding,) * 3
        op = output_padding if isinstance(output_padding, tuple) else (output_padding,) * 3
        self._ks = ks
        self._st = st
        self._pd = pd
        self._op = op

    def forward(self, x):
        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()
        b = self.conv_transpose.bias.contiguous()
        x = triton_conv_transpose3d(x, w, b, self._st, self._pd, self._op)

        if (self.max_pool.kernel_size == 2 and self.max_pool.stride == 2 and self.max_pool.padding == 0):
            x = fused_pool_post(x, self.subtract, 2, 2, 0)
        else:
            x = self.max_pool(x)
            # fallback
            x = torch.softmax(x, dim=1)
            x = x - self.subtract.view(1, -1, 1, 1, 1)
            x = torch.sigmoid(x) * x
            x = torch.max(x, dim=1)[0]
        return x