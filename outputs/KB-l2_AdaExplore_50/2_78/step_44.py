import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 4}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_W': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 8}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_W': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_W': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 16}, num_warps=8, num_stages=2),
    ],
    key=['N', 'C', 'D', 'H', 'W', 'OW'],
)
@triton.jit
def fused_pool_sum_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_on, stride_od, stride_oh, stride_ow,
    BLOCK_W: tl.constexpr,
):
    # one program per (n, od, oh) handles all ow in BLOCK_W chunks
    pid = tl.program_id(0)
    pid_w = tl.program_id(1)

    n = pid // (OD * OH)
    rem = pid % (OD * OH)
    od = rem // OH
    oh = rem % OH

    ow_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    ow_mask = ow_offs < OW

    # pooling: 2x2x2 then 3x3x3 = 6x6x6 window with stride 6
    # input window starts at (od*6, oh*6, ow*6), size 6
    d_start = od * 6
    h_start = oh * 6
    w_start = ow_offs * 6  # [BLOCK_W]

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    for c in range(0, C):
        # for each output element, max over 6x6x6 window
        # but actually it's max_pool(2) then max_pool(3) -- equivalent to max over 6x6x6 since both are non-overlapping
        max_val = tl.full([BLOCK_W], -float('inf'), dtype=tl.float32)
        for dd in range(0, 6):
            for hh in range(0, 6):
                for ww in range(0, 6):
                    d_idx = d_start + dd
                    h_idx = h_start + hh
                    w_idx = w_start + ww
                    in_mask = ow_mask & (d_idx < D) & (h_idx < H) & (w_idx < W)
                    offs = (n * stride_xn + c * stride_xc +
                            d_idx * stride_xd + h_idx * stride_xh +
                            w_idx * stride_xw)
                    v = tl.load(x_ptr + offs, mask=in_mask, other=-float('inf'))
                    max_val = tl.maximum(max_val, v)
        acc += max_val

    out_offs = (n * stride_on + od * stride_od + oh * stride_oh + ow_offs * stride_ow)
    tl.store(out_ptr + out_offs, acc, mask=ow_mask)


def fused_pool_sum(x):
    N, C, D, H, W = x.shape
    OD = D // 6
    OH = H // 6
    OW = W // 6
    out = torch.empty((N, 1, OD, OH, OW), device=x.device, dtype=x.dtype)

    grid = lambda meta: (N * OD * OH, triton.cdiv(OW, meta['BLOCK_W']))

    fused_pool_sum_kernel[grid](
        x, out,
        N, C, D, H, W,
        OD, OH, OW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(2), out.stride(3), out.stride(4),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)
        # Use channels_last_3d memory format for faster conv on modern GPUs
        self.conv_transpose = self.conv_transpose.to(memory_format=torch.channels_last_3d)

    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last_3d)
        x = self.conv_transpose(x)
        # fused: max_pool(2) -> max_pool(3) -> sum over channel
        return fused_pool_sum(x)