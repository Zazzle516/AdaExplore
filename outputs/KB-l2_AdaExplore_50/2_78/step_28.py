import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


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
    # pid covers (N * OD * OH * ceil(OW/BLOCK_OW))
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
            d_off = ch_off + d_in * stride_d  # [BLOCK_C]
            for hh in tl.static_range(6):
                h_in = h_base + hh
                h_valid = h_in < H
                dh_off = d_off + h_in * stride_h  # [BLOCK_C]
                for ww in tl.static_range(6):
                    w_in = w_base + ww  # [BLOCK_OW]
                    w_valid = (w_in < W) & ow_mask  # [BLOCK_OW]
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

    x = x.contiguous()
    out = torch.empty((N, 1, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OW = 16
    while BLOCK_OW < OW:
        BLOCK_OW *= 2
    if BLOCK_OW > 16:
        BLOCK_OW = 16  # cap

    ow_blocks = (OW + BLOCK_OW - 1) // BLOCK_OW
    grid = (N * OD * OH * ow_blocks,)
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
        BLOCK_OW=BLOCK_OW,
        num_warps=4,
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