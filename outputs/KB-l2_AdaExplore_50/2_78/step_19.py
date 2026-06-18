import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 16}, num_warps=2, num_stages=2),
    ],
    key=['N', 'C', 'D', 'H', 'W'],
)
@triton.jit
def fused_pool_sum_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    OHW = OH * OW
    ODHW = OD * OHW
    n = pid // ODHW
    rem = pid % ODHW
    od = rem // OHW
    rem2 = rem % OHW
    oh = rem2 // OW
    ow = rem2 % OW

    d_start = od * 6
    h_start = oh * 6
    w_start = ow * 6

    win = tl.arange(0, 216)
    dd = win // 36
    hh = (win // 6) % 6
    ww = win % 6

    HW = H * W
    DHW = D * HW
    base = n * C * DHW + (d_start + dd) * HW + (h_start + hh) * W + (w_start + ww)
    # base shape [216]

    acc = tl.zeros([1], dtype=tl.float32)
    NEG_INF = float('-inf')
    for c_start in range(0, C, BLOCK_C):
        cs = c_start + tl.arange(0, BLOCK_C)
        c_mask = cs < C
        offs = cs[:, None] * DHW + base[None, :]
        vals = tl.load(x_ptr + offs, mask=c_mask[:, None], other=NEG_INF)
        m = tl.max(vals, axis=1)
        m = tl.where(c_mask, m, 0.0)
        acc += tl.sum(m, axis=0)

    tl.store(out_ptr + pid, tl.sum(acc, axis=0))


def fused_pool_sum(x):
    N, C, D, H, W = x.shape
    OD = D // 6
    OH = H // 6
    OW = W // 6
    out = torch.empty((N, 1, OD, OH, OW), device=x.device, dtype=x.dtype)

    grid = (N * OD * OH * OW,)

    x_c = x.contiguous()
    fused_pool_sum_kernel[grid](
        x_c, out,
        N, C, D, H, W,
        OD, OH, OW,
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
        # fused: max_pool(2) -> max_pool(3) -> sum over channel
        return fused_pool_sum(x)