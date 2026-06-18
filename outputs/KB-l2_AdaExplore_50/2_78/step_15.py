import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


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
    tmp = pid // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    n = tmp // OD

    d_start = od * 6
    h_start = oh * 6
    w_start = ow * 6

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    max_vals = tl.full([BLOCK_C], -float('inf'), dtype=tl.float32)

    base = x_ptr + n * stride_xn + c_offs * stride_xc

    for dd in tl.static_range(0, 6):
        for hh in tl.static_range(0, 6):
            for ww in tl.static_range(0, 6):
                d_idx = d_start + dd
                h_idx = h_start + hh
                w_idx = w_start + ww
                ptrs = base + d_idx * stride_xd + h_idx * stride_xh + w_idx * stride_xw
                vals = tl.load(ptrs, mask=c_mask, other=-float('inf'))
                max_vals = tl.maximum(max_vals, vals)

    max_vals = tl.where(c_mask, max_vals, 0.0)
    s = tl.sum(max_vals, axis=0)

    out_ptr_off = out_ptr + n * stride_on + od * stride_od + oh * stride_oh + ow * stride_ow
    tl.store(out_ptr_off, s)


def fused_maxpool_maxpool_sum(x):
    N, C, D, H, W = x.shape
    OD = D // 6
    OH = H // 6
    OW = W // 6
    x = x.contiguous()
    out = torch.empty((N, 1, OD, OH, OW), dtype=x.dtype, device=x.device)

    BLOCK_C = triton.next_power_of_2(C)

    grid = (N * OD * OH * OW,)
    fused_pool_sum_kernel[grid](
        x, out,
        N, C, D, H, W,
        OD, OH, OW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(2), out.stride(3), out.stride(4),
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)
        self.conv_transpose.weight.data = self.conv_transpose.weight.data.to(memory_format=torch.channels_last_3d)

    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last_3d)
        x = self.conv_transpose(x)
        N, C, D, H, W = x.shape
        if D % 6 == 0 and H % 6 == 0 and W % 6 == 0:
            x = x.contiguous()
            return fused_maxpool_maxpool_sum(x)
        else:
            x = self.max_pool1(x)
            x = self.max_pool2(x)
            x = torch.sum(x, dim=1, keepdim=True)
            return x