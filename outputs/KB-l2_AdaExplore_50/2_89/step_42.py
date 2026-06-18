import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=2),
    ],
    key=['N_total', 'C'],
)
@triton.jit
def fused_pool_softmax_sub_swish_max_kernel(
    x_ptr, sub_ptr, out_ptr,
    N, C,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    N_total,
    BLOCK_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs < N_total

    # decompose offs -> (n, pd, ph, pw)
    pw = offs % W_out
    tmp = offs // W_out
    ph = tmp % H_out
    tmp2 = tmp // H_out
    pd = tmp2 % D_out
    n = tmp2 // D_out

    d0 = pd * 2
    h0 = ph * 2
    w0 = pw * 2

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    stride_n = C * D_in * H_in * W_in
    stride_c = D_in * H_in * W_in
    stride_d = H_in * W_in
    stride_h = W_in

    # base [BLOCK_C, BLOCK_N]
    base = (n[None, :] * stride_n
            + offs_c[:, None] * stride_c
            + d0[None, :] * stride_d
            + h0[None, :] * stride_h
            + w0[None, :])
    mask = mask_c[:, None] & mask_n[None, :]

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

    # softmax over channels axis=0
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

    tl.store(out_ptr + offs, out_val, mask=mask_n)


def fused_pool_post(x, sub):
    N, C, D_in, H_in, W_in = x.shape
    D_out = D_in // 2
    H_out = H_in // 2
    W_out = W_in // 2
    x_c = x.contiguous()
    out = torch.empty((N, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    BLOCK_C = triton.next_power_of_2(C)
    N_total = N * D_out * H_out * W_out
    grid = lambda META: ((N_total + META['BLOCK_N'] - 1) // META['BLOCK_N'],)
    fused_pool_softmax_sub_swish_max_kernel[grid](
        x_c, sub.contiguous(), out,
        N, C,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        N_total,
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
        if (self.max_pool.kernel_size == 2 and self.max_pool.stride == 2 and self.max_pool.padding == 0):
            x = fused_pool_post(x, self.subtract)
        else:
            x = self.max_pool(x)
            # fallback - just do it in torch
            x = torch.softmax(x, dim=1)
            x = x - self.subtract.view(1, -1, 1, 1, 1)
            x = torch.sigmoid(x) * x
            x = torch.max(x, dim=1)[0]
        return x