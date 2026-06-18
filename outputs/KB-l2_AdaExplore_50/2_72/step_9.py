import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bn_avgpool4_kernel(
    x_ptr, scale_ptr, shift_ptr, out_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    BLOCK: tl.constexpr,
):
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

    d0 = od * 4
    h0 = oh * 4
    w0 = ow * 4

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    base = ((n * C + c) * D)
    for dd in tl.static_range(0, 4):
        for hh in tl.static_range(0, 4):
            for ww in tl.static_range(0, 4):
                d_idx = d0 + dd
                h_idx = h0 + hh
                w_idx = w0 + ww
                in_off = ((base + d_idx) * H + h_idx) * W + w_idx
                v = tl.load(x_ptr + in_off, mask=mask, other=0.0)
                acc += v

    acc = acc * (1.0 / 64.0)

    s = tl.load(scale_ptr + c, mask=mask, other=0.0)
    sh = tl.load(shift_ptr + c, mask=mask, other=0.0)
    acc = acc * s + sh

    tl.store(out_ptr + offs, acc, mask=mask)


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
        x = x.contiguous()
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias

        y = F.conv_transpose3d(x, weight, bias, stride=self.stride, padding=self.padding)

        bn = self.batch_norm
        if self.training:
            y = F.batch_norm(
                y, bn.running_mean, bn.running_var,
                bn.weight, bn.bias,
                training=True, momentum=bn.momentum, eps=bn.eps,
            )
            N, C, D, H, W = y.shape
            OD, OH, OW = D // 4, H // 4, W // 4
            out = torch.empty((N, C, OD, OH, OW), device=y.device, dtype=y.dtype)
            total = N * C * OD * OH * OW
            BLOCK = 256
            grid = ((total + BLOCK - 1) // BLOCK,)
            scale = torch.ones(C, device=y.device, dtype=y.dtype)
            shift = torch.zeros(C, device=y.device, dtype=y.dtype)
            fused_bn_avgpool4_kernel[grid](
                y, scale, shift, out,
                N, C, D, H, W,
                OD, OH, OW,
                BLOCK=BLOCK, num_warps=4, num_stages=2,
            )
            return out
        else:
            eps = bn.eps
            var = bn.running_var
            mean = bn.running_mean
            gamma = bn.weight
            beta = bn.bias
            scale = gamma / torch.sqrt(var + eps)
            shift = beta - mean * scale

            N, C, D, H, W = y.shape
            OD, OH, OW = D // 4, H // 4, W // 4
            out = torch.empty((N, C, OD, OH, OW), device=y.device, dtype=y.dtype)
            total = N * C * OD * OH * OW
            BLOCK = 512
            grid = ((total + BLOCK - 1) // BLOCK,)
            fused_bn_avgpool4_kernel[grid](
                y, scale.contiguous(), shift.contiguous(), out,
                N, C, D, H, W,
                OD, OH, OW,
                BLOCK=BLOCK, num_warps=4, num_stages=2,
            )
            return out