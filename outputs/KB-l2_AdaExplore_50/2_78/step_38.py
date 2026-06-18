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
    # one program per (n, od, oh, ow); reduce over C in tiles
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

    # offsets within the 6x6x6 window (use 8 = next pow2, mask to 6)
    OFF = tl.arange(0, 8)
    in_mask = OFF < 6  # [8]

    d_idx = d_start + OFF  # [8]
    h_idx = h_start + OFF
    w_idx = w_start + OFF

    # build a [8,8,8] offset grid for spatial window per channel
    # spatial offsets relative to (n, c=0)
    sp_off = (d_idx[:, None, None] * stride_xd +
              h_idx[None, :, None] * stride_xh +
              w_idx[None, None, :] * stride_xw)  # [8,8,8]
    sp_mask = (in_mask[:, None, None] & in_mask[None, :, None] & in_mask[None, None, :])

    base_n = n * stride_xn

    # accumulator over channels
    total = 0.0

    c_offs = tl.arange(0, BLOCK_C)
    for c_start in range(0, C, BLOCK_C):
        c_idx = c_start + c_offs  # [BLOCK_C]
        c_mask = c_idx < C
        # offsets [BLOCK_C, 8, 8, 8]
        offs = (base_n
                + c_idx[:, None, None, None] * stride_xc
                + sp_off[None, :, :, :])
        mask = c_mask[:, None, None, None] & sp_mask[None, :, :, :]
        v = tl.load(x_ptr + offs, mask=mask, other=-float('inf'))
        # max over spatial dims -> [BLOCK_C]
        m = tl.max(tl.max(tl.max(v, axis=3), axis=2), axis=1)
        # mask invalid channels to 0
        m = tl.where(c_mask, m, 0.0)
        total += tl.sum(m, axis=0)

    out_off = n * stride_on + od * stride_od + oh * stride_oh + ow * stride_ow
    tl.store(out_ptr + out_off, total)


def fused_pool_sum(x):
    N, C, D, H, W = x.shape
    OD = D // 6
    OH = H // 6
    OW = W // 6
    x = x.contiguous()
    out = torch.empty((N, 1, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_C = 64 if C >= 64 else triton.next_power_of_2(C)

    grid = (N * OD * OH * OW,)
    fused_pool_sum_kernel[grid](
        x, out,
        N, C, D, H, W,
        OD, OH, OW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(2), out.stride(3), out.stride(4),
        BLOCK_C=BLOCK_C,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)

    def forward(self, x):
        x = self.conv_transpose(x)
        return fused_pool_sum(x)