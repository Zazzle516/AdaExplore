import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 128}, num_warps=2),
        triton.Config({'BLOCK': 256}, num_warps=4),
        triton.Config({'BLOCK': 512}, num_warps=4),
        triton.Config({'BLOCK': 512}, num_warps=8),
        triton.Config({'BLOCK': 1024}, num_warps=8),
    ],
    key=['total'],
)
@triton.jit
def _bn_avgpool4_kernel(
    in_ptr, out_ptr,
    scale_ptr, shift_ptr,
    N, C, ID, IH, IW,
    OD, OH, OW,
    BLOCK: tl.constexpr,
):
    # one program per (n, c, od, oh tile of ow)
    pid = tl.program_id(0)
    total = N * C * OD * OH * OW
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    ow = offs % OW
    t1 = offs // OW
    oh = t1 % OH
    t2 = t1 // OH
    od = t2 % OD
    t3 = t2 // OD
    c = t3 % C
    n = t3 // C

    # avg pool kernel size 4, stride 4 (two avgpool2 stacked)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    scale = tl.load(scale_ptr + c, mask=mask, other=0.0)
    shift = tl.load(shift_ptr + c, mask=mask, other=0.0)

    for dd in range(0, 4):
        for hh in range(0, 4):
            for ww in range(0, 4):
                id_ = od * 4 + dd
                ih_ = oh * 4 + hh
                iw_ = ow * 4 + ww
                in_off = ((n * C + c) * ID + id_) * IH * IW + ih_ * IW + iw_
                v = tl.load(in_ptr + in_off, mask=mask, other=0.0)
                v = v * scale + shift
                acc = acc + v

    acc = acc / 64.0
    tl.store(out_ptr + offs, acc, mask=mask)


def fused_bn_avgpool(x, scale, shift):
    N, C, ID, IH, IW = x.shape
    OD, OH, OW = ID // 4, IH // 4, IW // 4
    x = x.contiguous()
    out = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=torch.float32)
    total = N * C * OD * OH * OW
    grid = lambda META: ((total + META['BLOCK'] - 1) // META['BLOCK'],)
    _bn_avgpool4_kernel[grid](
        x, out, scale.contiguous(), shift.contiguous(),
        N, C, ID, IH, IW, OD, OH, OW,
        total,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias

        y = F.conv_transpose3d(x, weight, bias, stride=self.stride, padding=self.padding)

        # batch norm fold
        if self.training:
            y = self.batch_norm(y)
            y = F.avg_pool3d(y, 2)
            y = F.avg_pool3d(y, 2)
            return y
        else:
            rm = self.batch_norm.running_mean
            rv = self.batch_norm.running_var
            eps = self.batch_norm.eps
            w = self.batch_norm.weight
            b = self.batch_norm.bias
            invstd = torch.rsqrt(rv + eps)
            scale = w * invstd
            shift = b - rm * scale
            return fused_bn_avgpool(y, scale, shift)