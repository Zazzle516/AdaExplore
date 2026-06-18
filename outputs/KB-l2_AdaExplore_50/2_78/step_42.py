import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_OW': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_C': 32, 'BLOCK_OW': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 32, 'BLOCK_OW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_OW': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_OW': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_C': 64, 'BLOCK_OW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_OW': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_OW': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_C': 32, 'BLOCK_OW': 32}, num_warps=8, num_stages=2),
    ],
    key=['N', 'C', 'D', 'H', 'W'],
)
@triton.jit
def fused_pool_sum_kernel(
    in_ptr, out_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    BLOCK_C: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    pid = tl.program_id(0)
    ow_blocks = (OW + BLOCK_OW - 1) // BLOCK_OW
    ow_blk = pid % ow_blocks
    tmp = pid // ow_blocks
    oh = tmp % OH
    tmp2 = tmp // OH
    od = tmp2 % OD
    n = tmp2 // OD

    ow_offs = ow_blk * BLOCK_OW + tl.arange(0, BLOCK_OW)
    ow_mask = ow_offs < OW

    d_base = od * 6
    h_base = oh * 6
    w_base = ow_offs * 6  # [BLOCK_OW]

    c_offs = tl.arange(0, BLOCK_C)
    acc = tl.zeros([BLOCK_OW], dtype=tl.float32)

    n_off = n * stride_n

    for c_start in range(0, C, BLOCK_C):
        c_idx = c_start + c_offs
        c_mask = c_idx < C
        ch_off = n_off + c_idx * stride_c  # [BLOCK_C]

        ch_max = tl.full([BLOCK_C, BLOCK_OW], -float('inf'), dtype=tl.float32)

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
                    w_valid = (w_in < W) & ow_mask
                    offset = dh_off[:, None] + w_in[None, :] * stride_w
                    in_bounds = (c_mask[:, None] & (d_valid & h_valid)) & w_valid[None, :]
                    val = tl.load(in_ptr + offset, mask=in_bounds, other=-float('inf'))
                    ch_max = tl.maximum(ch_max, val)

        ch_max = tl.where(c_mask[:, None], ch_max, 0.0)
        acc += tl.sum(ch_max, axis=0)

    out_base = ((n * OD + od) * OH + oh) * OW
    tl.store(out_ptr + out_base + ow_offs, acc, mask=ow_mask)


def fused_pool_sum(x):
    N, C, D, H, W = x.shape
    D1 = D // 2
    H1 = H // 2
    W1 = W // 2
    OD = D1 // 3
    OH = H1 // 3
    OW = W1 // 3

    # channels_last_3d: makes C the contiguous axis, good for our inner C loop
    x = x.to(memory_format=torch.channels_last_3d)
    out = torch.empty((N, 1, OD, OH, OW), device=x.device, dtype=x.dtype)

    stride_n, stride_c, stride_d, stride_h, stride_w = x.stride()

    grid = lambda meta: (N * OD * OH * ((OW + meta['BLOCK_OW'] - 1) // meta['BLOCK_OW']),)

    fused_pool_sum_kernel[grid](
        x, out,
        N, C, D, H, W,
        OD, OH, OW,
        stride_n, stride_c, stride_d, stride_h, stride_w,
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