import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_softmax_sub_swish_max_kernel(
    x_ptr, sub_ptr, out_ptr,
    N, C, S,
    BLOCK_C: tl.constexpr,
):
    # one program per (n, s)
    pid = tl.program_id(0)
    n = pid // S
    s = pid % S

    offs = tl.arange(0, BLOCK_C)
    mask = offs < C

    base = n * C * S + s
    x = tl.load(x_ptr + base + offs * S, mask=mask, other=-float('inf'))

    # softmax over channels
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    z = tl.sum(e, axis=0)
    sm = e / z

    sub = tl.load(sub_ptr + offs, mask=mask, other=0.0)
    y = sm - sub
    # swish: y * sigmoid(y)
    sw = y * tl.sigmoid(y)
    sw = tl.where(mask, sw, -float('inf'))
    out_val = tl.max(sw, axis=0)

    tl.store(out_ptr + n * S + s, out_val)


@triton.jit
def fused_pool_softmax_sub_swish_max_kernel(
    x_ptr, sub_ptr, out_ptr,
    N, C,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    BLOCK_C: tl.constexpr,
):
    # one program per output (n, pd, ph, pw)
    pid = tl.program_id(0)
    S_out = D_out * H_out * W_out
    n = pid // S_out
    rem = pid % S_out
    pd = rem // (H_out * W_out)
    rem2 = rem % (H_out * W_out)
    ph = rem2 // W_out
    pw = rem2 % W_out

    d0 = pd * 2
    h0 = ph * 2
    w0 = pw * 2

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    # base index for (n, c, d0, h0, w0): n*C*D*H*W + c*D*H*W + ...
    stride_n = C * D_in * H_in * W_in
    stride_c = D_in * H_in * W_in
    stride_d = H_in * W_in
    stride_h = W_in

    base = n * stride_n + offs_c * stride_c + d0 * stride_d + h0 * stride_h + w0

    # 8 values for 2x2x2 pool
    v000 = tl.load(x_ptr + base, mask=mask_c, other=-float('inf'))
    v001 = tl.load(x_ptr + base + 1, mask=mask_c, other=-float('inf'))
    v010 = tl.load(x_ptr + base + stride_h, mask=mask_c, other=-float('inf'))
    v011 = tl.load(x_ptr + base + stride_h + 1, mask=mask_c, other=-float('inf'))
    v100 = tl.load(x_ptr + base + stride_d, mask=mask_c, other=-float('inf'))
    v101 = tl.load(x_ptr + base + stride_d + 1, mask=mask_c, other=-float('inf'))
    v110 = tl.load(x_ptr + base + stride_d + stride_h, mask=mask_c, other=-float('inf'))
    v111 = tl.load(x_ptr + base + stride_d + stride_h + 1, mask=mask_c, other=-float('inf'))

    x = tl.maximum(tl.maximum(tl.maximum(v000, v001), tl.maximum(v010, v011)),
                   tl.maximum(tl.maximum(v100, v101), tl.maximum(v110, v111)))

    # softmax over channels
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask_c, e, 0.0)
    z = tl.sum(e, axis=0)
    sm = e / z

    sub = tl.load(sub_ptr + offs_c, mask=mask_c, other=0.0)
    y = sm - sub
    sw = y * tl.sigmoid(y)
    sw = tl.where(mask_c, sw, -float('inf'))
    out_val = tl.max(sw, axis=0)

    tl.store(out_ptr + n * S_out + rem, out_val)


def fused_pool_post(x, sub, pool_k, pool_s, pool_p):
    # x: post-conv (N, C, D, H, W); pool 2x2x2 stride 2 padding 0
    N, C, D_in, H_in, W_in = x.shape
    D_out = (D_in - pool_k) // pool_s + 1
    H_out = (H_in - pool_k) // pool_s + 1
    W_out = (W_in - pool_k) // pool_s + 1
    x_c = x.contiguous()
    out = torch.empty((N, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    BLOCK_C = triton.next_power_of_2(C)
    grid = (N * D_out * H_out * W_out,)
    fused_pool_softmax_sub_swish_max_kernel[grid](
        x_c, sub.contiguous(), out,
        N, C,
        D_in, H_in, W_in,
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
        # Fuse MaxPool3d + softmax + subtract + swish + channel-max
        if (self.max_pool.kernel_size == 2 and self.max_pool.stride == 2 and self.max_pool.padding == 0):
            x = fused_pool_post(x, self.subtract, 2, 2, 0)
        else:
            x = self.max_pool(x)
            x = fused_post(x, self.subtract)
        return x