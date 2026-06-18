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
    # one program per (n, od, oh, ow)
    pid = tl.program_id(0)
    ow = pid % OW
    pid2 = pid // OW
    oh = pid2 % OH
    pid3 = pid2 // OH
    od = pid3 % OD
    n = pid3 // OD

    # The fused pool is maxpool2 (k=2) then maxpool3 (k=3) -> effective pool k=6 stride=6
    # Output element (od, oh, ow) corresponds to input region:
    # In maxpool2 output coords: [od*3 : od*3+3] etc, each of those is max of 2x2x2 in original
    # Equivalently, max over a 6x6x6 window with stride 6 in original (after conv) tensor
    d_start = od * 6
    h_start = oh * 6
    w_start = ow * 6

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    # accumulator for sum over channels of max
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    max_vals = tl.full([BLOCK_C], -float('inf'), dtype=tl.float32)

    for dd in tl.static_range(0, 6):
        for hh in tl.static_range(0, 6):
            for ww in tl.static_range(0, 6):
                d_idx = d_start + dd
                h_idx = h_start + hh
                w_idx = w_start + ww
                ptrs = x_ptr + n * stride_xn + c_offs * stride_xc + d_idx * stride_xd + h_idx * stride_xh + w_idx * stride_xw
                vals = tl.load(ptrs, mask=c_mask, other=-float('inf'))
                max_vals = tl.maximum(max_vals, vals)

    # sum across channels
    summed = tl.sum(tl.where(c_mask, max_vals, 0.0), axis=0)

    out_ptr_off = out_ptr + n * stride_on + od * stride_od + oh * stride_oh + ow * stride_ow
    tl.store(out_ptr_off, summed)


def fused_pool_sum(x, OD, OH, OW):
    # x: (N, C, D, H, W) - output of conv_transpose
    # Effective: maxpool(k=2) -> maxpool(k=3) -> sum(dim=1, keepdim=True)
    N, C, D, H, W = x.shape
    x = x.contiguous()
    out = torch.empty((N, 1, OD, OH, OW), device=x.device, dtype=x.dtype)

    # find next pow2 >= C
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

    def forward(self, x):
        x = self.conv_transpose(x)
        # The fused pool computes max over a 6x6x6 window at (od*6, oh*6, ow*6)
        # which matches maxpool(k=2,s=2) -> maxpool(k=3,s=3) using floor division.
        N, C, D, H, W = x.shape
        OD = (D // 2) // 3
        OH = (H // 2) // 3
        OW = (W // 2) // 3
        if OD * 6 <= D and OH * 6 <= H and OW * 6 <= W:
            return fused_pool_sum(x, OD, OH, OW)
        else:
            x = self.max_pool1(x)
            x = self.max_pool2(x)
            x = torch.sum(x, dim=1, keepdim=True)
            return x