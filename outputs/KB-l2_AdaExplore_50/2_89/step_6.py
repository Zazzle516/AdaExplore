import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_pool_softmax_sub_swish_max_kernel(
    x_ptr, sub_ptr, out_ptr,
    N, C,
    Di, Hi, Wi,       # conv output spatial dims
    Do, Ho, Wo,       # pooled output spatial dims
    PK: tl.constexpr, # pool kernel size (assumed cubic)
    PS: tl.constexpr, # pool stride
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    # decode (n, dp, hp, wp)
    SO = Do * Ho * Wo
    n = pid // SO
    rem = pid % SO
    dp = rem // (Ho * Wo)
    rem2 = rem % (Ho * Wo)
    hp = rem2 // Wo
    wp = rem2 % Wo

    d0 = dp * PS
    h0 = hp * PS
    w0 = wp * PS

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    SI = Di * Hi * Wi
    base = n * C * SI + offs_c * SI

    # Compute maxpool over PK^3 window for each channel
    neg_inf = float('-inf')
    max_val = tl.full([BLOCK_C], neg_inf, dtype=tl.float32)
    for kd in tl.static_range(0, PK):
        for kh in tl.static_range(0, PK):
            for kw in tl.static_range(0, PK):
                d = d0 + kd
                h = h0 + kh
                w = w0 + kw
                in_bounds = (d < Di) & (h < Hi) & (w < Wi)
                offs = base + d * Hi * Wi + h * Wi + w
                v = tl.load(x_ptr + offs, mask=mask_c & in_bounds, other=neg_inf)
                max_val = tl.maximum(max_val, v)

    x = tl.where(mask_c, max_val, neg_inf)

    # softmax over channels
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask_c, e, 0.0)
    z = tl.sum(e, axis=0)
    sm = e / z

    sub = tl.load(sub_ptr + offs_c, mask=mask_c, other=0.0)
    y = sm - sub
    sw = y * tl.sigmoid(y)
    sw = tl.where(mask_c, sw, neg_inf)
    out_val = tl.max(sw, axis=0)

    tl.store(out_ptr + n * SO + rem, out_val)


def fused_post(x, sub, pool_kernel, pool_stride):
    # x: (N, C, Di, Hi, Wi) - conv output (pre-pool)
    N, C, Di, Hi, Wi = x.shape
    Do = (Di - pool_kernel) // pool_stride + 1
    Ho = (Hi - pool_kernel) // pool_stride + 1
    Wo = (Wi - pool_kernel) // pool_stride + 1
    x_c = x if x.is_contiguous() else x.contiguous()
    out = torch.empty((N, Do, Ho, Wo), device=x.device, dtype=x.dtype)
    BLOCK_C = triton.next_power_of_2(C)
    grid = (N * Do * Ho * Wo,)
    fused_pool_softmax_sub_swish_max_kernel[grid](
        x_c, sub.contiguous(), out,
        N, C,
        Di, Hi, Wi,
        Do, Ho, Wo,
        PK=pool_kernel, PS=pool_stride,
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
        x = fused_post(x, self.subtract, self.max_pool.kernel_size if isinstance(self.max_pool.kernel_size, int) else self.max_pool.kernel_size[0],
                       self.max_pool.stride if isinstance(self.max_pool.stride, int) else self.max_pool.stride[0])
        return x