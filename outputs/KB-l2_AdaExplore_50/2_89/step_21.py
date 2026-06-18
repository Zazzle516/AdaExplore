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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 4}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_W': 8}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_W': 16}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_W': 8}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_W': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_W': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 16}, num_warps=4, num_stages=2),
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
    # one program processes BLOCK_W output points along W_out
    pid = tl.program_id(0)
    W_tiles = (W_out + BLOCK_W - 1) // BLOCK_W
    rows = N * D_out * H_out
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
    w0 = offs_w * 2  # [BLOCK_W]

    stride_n = C * D_in * H_in * W_in
    stride_c = D_in * H_in * W_in
    stride_d = H_in * W_in
    stride_h = W_in

    # base shape: [BLOCK_C, BLOCK_W]
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

    # softmax over channels (axis=0)
    m = tl.max(x, axis=0)  # [BLOCK_W]
    e = tl.exp(x - m[None, :])
    e = tl.where(mask_c[:, None], e, 0.0)
    z = tl.sum(e, axis=0)  # [BLOCK_W]
    sm = e / z[None, :]

    sub = tl.load(sub_ptr + offs_c, mask=mask_c, other=0.0)  # [BLOCK_C]
    y = sm - sub[:, None]
    sw = y * tl.sigmoid(y)
    sw = tl.where(mask_c[:, None], sw, -float('inf'))
    out_val = tl.max(sw, axis=0)  # [BLOCK_W]

    S_out = D_out * H_out * W_out
    out_off = n * S_out + pd * H_out * W_out + ph * W_out + offs_w
    tl.store(out_ptr + out_off, out_val, mask=mask_w)


def fused_pool_post(x, sub, pool_k, pool_s, pool_p):
    # x: post-conv (N, C, D, H, W); pool 2x2x2 stride 2 padding 0
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

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fuse MaxPool3d + softmax + subtract + swish + channel-max
        if (self.max_pool.kernel_size == 2 and self.max_pool.stride == 2 and self.max_pool.padding == 0):
            x = fused_pool_post(x, self.subtract, 2, 2, 0)
        else:
            x = self.max_pool(x)
            x = fused_post(x, self.subtract)
        return x