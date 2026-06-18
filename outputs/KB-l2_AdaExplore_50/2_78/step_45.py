import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
    ],
    key=['C', 'D', 'H', 'W'],
)
@triton.jit
def fused_pool_sum_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    BLOCK_C: tl.constexpr,
):
    # one program per (n, od, oh, ow)
    pid = tl.program_id(0)
    ow = pid % OW
    tmp = pid // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    n = tmp // OD

    # After max_pool1 (k=2) then max_pool2 (k=3), effective window per output element is 6x6x6
    # output[od,oh,ow] = max over d in [6*od, 6*od+6), similarly h, w
    d_start = od * 6
    h_start = oh * 6
    w_start = ow * 6

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    # accumulator for max per channel
    neg_inf = float('-inf')
    max_vals = tl.full((BLOCK_C,), neg_inf, dtype=tl.float32)

    base = n * stride_n

    for dd in tl.static_range(0, 6):
        for hh in tl.static_range(0, 6):
            for ww in tl.static_range(0, 6):
                d_idx = d_start + dd
                h_idx = h_start + hh
                w_idx = w_start + ww
                in_bounds = (d_idx < D) & (h_idx < H) & (w_idx < W)
                offs = base + c_offs * stride_c + d_idx * stride_d + h_idx * stride_h + w_idx * stride_w
                mask = c_mask & in_bounds
                vals = tl.load(x_ptr + offs, mask=mask, other=neg_inf)
                max_vals = tl.maximum(max_vals, vals)

    # sum over channels
    s = tl.sum(tl.where(c_mask, max_vals, 0.0), axis=0)

    out_off = ((n * OD + od) * OH + oh) * OW + ow
    tl.store(out_ptr + out_off, s)


def fused_pool_sum(x):
    x = x.contiguous()
    N, C, D, H, W = x.shape
    OD = D // 6
    OH = H // 6
    OW = W // 6
    out = torch.empty((N, 1, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_C = triton.next_power_of_2(C)
    grid = (N * OD * OH * OW,)
    fused_pool_sum_kernel[grid](
        x, out,
        N, C, D, H, W,
        OD, OH, OW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.conv_transpose = self.conv_transpose.to(memory_format=torch.channels_last_3d)

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last_3d)
        x = self.conv_transpose(x)
        x = fused_pool_sum(x)
        return x