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
    pid_nc = tl.program_id(0)
    pid_s = tl.program_id(1)
    n = pid_nc // C
    c = pid_nc % C

    out_spatial = OD * OH * OW
    offs = pid_s * BLOCK + tl.arange(0, BLOCK)
    mask = offs < out_spatial

    ow = offs % OW
    t1 = offs // OW
    oh = t1 % OH
    od = t1 // OH

    d0 = od * 4
    h0 = oh * 4
    w0 = ow * 4

    acc = tl.zeros((BLOCK,), dtype=tl.float32)

    base = (n * C + c) * D
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

    s = tl.load(scale_ptr + c)
    sh = tl.load(shift_ptr + c)
    acc = acc * s + sh

    out_base = (n * C + c) * out_spatial
    tl.store(out_ptr + out_base + offs, acc, mask=mask)


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
            N, C, D, H, W = y.shape
            y_flat = y.reshape(N, C, -1)
            mean = y_flat.mean(dim=(0, 2))
            var_biased = y_flat.var(dim=(0, 2), unbiased=False)
            with torch.no_grad():
                n_elem = N * D * H * W
                if n_elem > 1:
                    var_unbiased = var_biased * (n_elem / (n_elem - 1))
                else:
                    var_unbiased = var_biased
                bn.running_mean.mul_(1 - bn.momentum).add_(mean.detach(), alpha=bn.momentum)
                bn.running_var.mul_(1 - bn.momentum).add_(var_unbiased.detach(), alpha=bn.momentum)
                bn.num_batches_tracked.add_(1)
            scale = bn.weight / torch.sqrt(var_biased + bn.eps)
            shift = bn.bias - mean * scale
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
        out_spatial = OD * OH * OW
        BLOCK = 256
        grid = (N * C, (out_spatial + BLOCK - 1) // BLOCK)
        fused_bn_avgpool4_kernel[grid](
            y, scale.contiguous(), shift.contiguous(), out,
            N, C, D, H, W,
            OD, OH, OW,
            BLOCK=BLOCK, num_warps=4, num_stages=2,
        )
        return out