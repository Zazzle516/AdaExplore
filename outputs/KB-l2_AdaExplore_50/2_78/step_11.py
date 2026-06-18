import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_pool_sum_kernel(
    in_ptr, out_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    ow = pid % OW
    tmp = pid // OW
    oh = tmp % OH
    tmp2 = tmp // OH
    od = tmp2 % OD
    n = tmp2 // OD

    d_base = od * 6
    h_base = oh * 6
    w_base = ow * 6

    c_offs = tl.arange(0, BLOCK_C)
    acc = tl.zeros([], dtype=tl.float32)

    n_off = n * stride_n

    for c_start in range(0, C, BLOCK_C):
        c_idx = c_start + c_offs
        c_mask = c_idx < C
        ch_off = n_off + c_idx * stride_c

        ch_max = tl.full([BLOCK_C], -float('inf'), dtype=tl.float32)

        # Unrolled 6x6x6 window
        for dd in tl.static_range(6):
            d_in = d_base + dd
            d_valid = d_in < D
            d_off = ch_off + d_in * stride_d
            for hh in tl.static_range(6):
                h_in = h_base + hh
                h_valid = h_in < H
                dh_off = d_off + h_in * stride_h
                for ww in tl.static_range(6):
                    w_in = w_base + ww
                    w_valid = w_in < W
                    in_bounds = d_valid & h_valid & w_valid
                    offset = dh_off + w_in * stride_w
                    val = tl.load(in_ptr + offset, mask=c_mask & in_bounds, other=-float('inf'))
                    ch_max = tl.maximum(ch_max, val)

        ch_max = tl.where(c_mask, ch_max, 0.0)
        acc += tl.sum(ch_max, axis=0)

    out_off = ((n * OD + od) * OH + oh) * OW + ow
    tl.store(out_ptr + out_off, acc)


def fused_pool_sum(x):
    N, C, D, H, W = x.shape
    D1 = D // 2
    H1 = H // 2
    W1 = W // 2
    OD = D1 // 3
    OH = H1 // 3
    OW = W1 // 3

    x = x.contiguous()
    out = torch.empty((N, 1, OD, OH, OW), device=x.device, dtype=x.dtype)

    grid = (N * OD * OH * OW,)
    BLOCK_C = 64
    if C <= 32:
        BLOCK_C = 32
    if C <= 16:
        BLOCK_C = 16

    stride_n, stride_c, stride_d, stride_h, stride_w = x.stride()

    fused_pool_sum_kernel[grid](
        x, out,
        N, C, D, H, W,
        OD, OH, OW,
        stride_n, stride_c, stride_d, stride_h, stride_w,
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
        x = fused_pool_sum(x)
        return x