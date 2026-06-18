import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@triton.jit
def fused_pool_sum_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_on, stride_od, stride_oh, stride_ow,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    ow = pid % OW
    pid2 = pid // OW
    oh = pid2 % OH
    pid3 = pid2 // OH
    od = pid3 % OD
    n = pid3 // OD

    d_start = od * 6
    h_start = oh * 6
    w_start = ow * 6

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    max_vals = tl.full([BLOCK_C], -float('inf'), dtype=tl.float32)

    base = x_ptr + n * stride_xn + c_offs * stride_xc

    for dd in tl.static_range(0, 6):
        for hh in tl.static_range(0, 6):
            row_base = base + (d_start + dd) * stride_xd + (h_start + hh) * stride_xh + w_start * stride_xw
            v0 = tl.load(row_base + 0 * stride_xw, mask=c_mask, other=-float('inf'))
            v1 = tl.load(row_base + 1 * stride_xw, mask=c_mask, other=-float('inf'))
            v2 = tl.load(row_base + 2 * stride_xw, mask=c_mask, other=-float('inf'))
            v3 = tl.load(row_base + 3 * stride_xw, mask=c_mask, other=-float('inf'))
            v4 = tl.load(row_base + 4 * stride_xw, mask=c_mask, other=-float('inf'))
            v5 = tl.load(row_base + 5 * stride_xw, mask=c_mask, other=-float('inf'))
            m01 = tl.maximum(v0, v1)
            m23 = tl.maximum(v2, v3)
            m45 = tl.maximum(v4, v5)
            m = tl.maximum(tl.maximum(m01, m23), m45)
            max_vals = tl.maximum(max_vals, m)

    summed = tl.sum(tl.where(c_mask, max_vals, 0.0), axis=0)

    out_ptr_off = out_ptr + n * stride_on + od * stride_od + oh * stride_oh + ow * stride_ow
    tl.store(out_ptr_off, summed)


def fused_pool_sum(x):
    N, C, D, H, W = x.shape
    OD = (D // 2) // 3
    OH = (H // 2) // 3
    OW = (W // 2) // 3
    x = x.contiguous()
    out = torch.empty((N, 1, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    grid = (N * OD * OH * OW,)
    fused_pool_sum_kernel[grid](
        x, out,
        N, C, D, H, W,
        OD, OH, OW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(2), out.stride(3), out.stride(4),
        BLOCK_C=BLOCK_C,
        num_warps=2,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)
        try:
            self.conv_transpose.weight.data = self.conv_transpose.weight.data.contiguous(memory_format=torch.channels_last_3d)
        except Exception:
            pass

    def forward(self, x):
        if x.is_cuda:
            try:
                x = x.contiguous(memory_format=torch.channels_last_3d)
            except Exception:
                x = x.contiguous()
        else:
            x = x.contiguous()
        x = self.conv_transpose(x)
        x = x.contiguous()
        N, C, D, H, W = x.shape
        if (D // 2) // 3 >= 1 and (H // 2) // 3 >= 1 and (W // 2) // 3 >= 1:
            return fused_pool_sum(x)
        else:
            x = self.max_pool1(x)
            x = self.max_pool2(x)
            x = torch.sum(x, dim=1, keepdim=True)
            return x