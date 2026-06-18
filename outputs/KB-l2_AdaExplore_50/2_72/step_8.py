import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 128}, num_warps=2),
        triton.Config({'BLOCK': 128}, num_warps=4),
        triton.Config({'BLOCK': 256}, num_warps=2),
        triton.Config({'BLOCK': 256}, num_warps=4),
        triton.Config({'BLOCK': 256}, num_warps=8),
        triton.Config({'BLOCK': 512}, num_warps=4),
        triton.Config({'BLOCK': 512}, num_warps=8),
        triton.Config({'BLOCK': 1024}, num_warps=8),
    ],
    key=['OUT_PER_NC', 'D', 'H', 'W'],
)
@triton.jit
def fused_bn_avgpool4_kernel(
    x_ptr, out_ptr,
    scale_ptr, shift_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    OUT_PER_NC,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    BLOCK: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_t = tl.program_id(1)

    n = pid_nc // C
    c = pid_nc % C

    scale = tl.load(scale_ptr + c)
    shift = tl.load(shift_ptr + c)

    offs = pid_t * BLOCK + tl.arange(0, BLOCK)
    mask = offs < OUT_PER_NC

    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    od = tmp // OH

    d0 = od * 4
    h0 = oh * 4
    w0 = ow * 4

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    base = n * stride_n + c * stride_c

    for dd in tl.static_range(0, 4):
        for hh in tl.static_range(0, 4):
            for ww in tl.static_range(0, 4):
                d_idx = d0 + dd
                h_idx = h0 + hh
                w_idx = w0 + ww
                in_bounds = mask & (d_idx < D) & (h_idx < H) & (w_idx < W)
                ptr = base + d_idx * stride_d + h_idx * stride_h + w_idx * stride_w
                v = tl.load(x_ptr + ptr, mask=in_bounds, other=0.0)
                acc += v

    acc = acc * (scale / 64.0) + shift

    out_ptr_off = (n * out_stride_n + c * out_stride_c +
                   od * out_stride_d + oh * out_stride_h + ow * out_stride_w)
    tl.store(out_ptr + out_ptr_off, acc, mask=mask)


def fused_bn_avgpool4(x, scale, shift):
    N, C, D, H, W = x.shape
    OD = D // 4
    OH = H // 4
    OW = W // 4
    out = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)
    OUT_PER_NC = OD * OH * OW
    grid = lambda meta: (N * C, (OUT_PER_NC + meta['BLOCK'] - 1) // meta['BLOCK'])
    fused_bn_avgpool4_kernel[grid](
        x, out, scale, shift,
        N, C, D, H, W, OD, OH, OW,
        OUT_PER_NC,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.avg_pool1 = nn.AvgPool3d(kernel_size=2)
        self.avg_pool2 = nn.AvgPool3d(kernel_size=2)

    def forward(self, x):
        x = self.conv_transpose(x)

        if self.training:
            x = self.batch_norm(x)
            x = self.avg_pool1(x)
            x = self.avg_pool2(x)
            return x
        else:
            # Fold BN affine + running stats into scale/shift
            bn = self.batch_norm
            rm = bn.running_mean
            rv = bn.running_var
            eps = bn.eps
            w = bn.weight
            b = bn.bias
            invstd = torch.rsqrt(rv + eps)
            scale = w * invstd
            shift = b - rm * scale
            x = x.contiguous()
            return fused_bn_avgpool4(x, scale.contiguous(), shift.contiguous())