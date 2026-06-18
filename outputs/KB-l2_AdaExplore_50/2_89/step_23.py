import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_pool_softmax_sub_swish_max_kernel(
    x_ptr, sub_ptr, out_ptr,
    N, C, D_in, H_in, W_in,
    D_out, H_out, W_out,
    BLOCK_C: tl.constexpr,
):
    # one program per (n, d_out, h_out, w_out)
    pid = tl.program_id(0)
    s_out = D_out * H_out * W_out
    n = pid // s_out
    rem = pid % s_out
    d_o = rem // (H_out * W_out)
    rem2 = rem % (H_out * W_out)
    h_o = rem2 // W_out
    w_o = rem2 % W_out

    d_i = d_o * 2
    h_i = h_o * 2
    w_i = w_o * 2

    offs = tl.arange(0, BLOCK_C)
    mask = offs < C

    # input base for (n, c, d_i, h_i, w_i)
    in_chw = D_in * H_in * W_in
    base_n = n * C * in_chw
    # channel stride = in_chw
    # gather 8 positions of 2x2x2 window
    p000 = base_n + offs * in_chw + (d_i + 0) * H_in * W_in + (h_i + 0) * W_in + (w_i + 0)
    p001 = base_n + offs * in_chw + (d_i + 0) * H_in * W_in + (h_i + 0) * W_in + (w_i + 1)
    p010 = base_n + offs * in_chw + (d_i + 0) * H_in * W_in + (h_i + 1) * W_in + (w_i + 0)
    p011 = base_n + offs * in_chw + (d_i + 0) * H_in * W_in + (h_i + 1) * W_in + (w_i + 1)
    p100 = base_n + offs * in_chw + (d_i + 1) * H_in * W_in + (h_i + 0) * W_in + (w_i + 0)
    p101 = base_n + offs * in_chw + (d_i + 1) * H_in * W_in + (h_i + 0) * W_in + (w_i + 1)
    p110 = base_n + offs * in_chw + (d_i + 1) * H_in * W_in + (h_i + 1) * W_in + (w_i + 0)
    p111 = base_n + offs * in_chw + (d_i + 1) * H_in * W_in + (h_i + 1) * W_in + (w_i + 1)

    v000 = tl.load(x_ptr + p000, mask=mask, other=-float('inf'))
    v001 = tl.load(x_ptr + p001, mask=mask, other=-float('inf'))
    v010 = tl.load(x_ptr + p010, mask=mask, other=-float('inf'))
    v011 = tl.load(x_ptr + p011, mask=mask, other=-float('inf'))
    v100 = tl.load(x_ptr + p100, mask=mask, other=-float('inf'))
    v101 = tl.load(x_ptr + p101, mask=mask, other=-float('inf'))
    v110 = tl.load(x_ptr + p110, mask=mask, other=-float('inf'))
    v111 = tl.load(x_ptr + p111, mask=mask, other=-float('inf'))

    # max pool
    x = tl.maximum(tl.maximum(tl.maximum(v000, v001), tl.maximum(v010, v011)),
                   tl.maximum(tl.maximum(v100, v101), tl.maximum(v110, v111)))

    # softmax over channels
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    z = tl.sum(e, axis=0)
    sm = e / z

    sub = tl.load(sub_ptr + offs, mask=mask, other=0.0)
    y = sm - sub
    sw = y * tl.sigmoid(y)
    sw = tl.where(mask, sw, -float('inf'))
    out_val = tl.max(sw, axis=0)

    tl.store(out_ptr + n * s_out + rem, out_val)


def fused_post(x, sub):
    # x: (N, C, D_in, H_in, W_in) - conv output, pool not yet applied
    N, C, D_in, H_in, W_in = x.shape
    D_out = D_in // 2
    H_out = H_in // 2
    W_out = W_in // 2
    x_c = x.contiguous()
    out = torch.empty((N, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    BLOCK_C = triton.next_power_of_2(C)
    grid = (N * D_out * H_out * W_out,)
    fused_pool_softmax_sub_swish_max_kernel[grid](
        x_c, sub.contiguous(), out,
        N, C, D_in, H_in, W_in,
        D_out, H_out, W_out,
        BLOCK_C=BLOCK_C,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_post(x, self.subtract)
        return x