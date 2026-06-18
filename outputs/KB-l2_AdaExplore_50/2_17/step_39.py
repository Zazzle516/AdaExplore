import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3x3_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W, OC, OH, OW,
    BLOCK_M: tl.constexpr,   # output spatial tile
    BLOCK_N: tl.constexpr,   # OC tile
    BLOCK_K: tl.constexpr,   # IC tile
):
    pid_n = tl.program_id(0)        # batch
    pid_oc = tl.program_id(1)       # OC tile
    pid_sp = tl.program_id(2)       # output-spatial tile

    OHW = OH * OW
    sp_offs = pid_sp * BLOCK_M + tl.arange(0, BLOCK_M)
    sp_mask = sp_offs < OHW
    oh = sp_offs // OW
    ow = sp_offs - oh * OW

    oc_offs = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # iterate over IC in tiles
    for ic0 in range(0, IC, BLOCK_K):
        ic_idx = ic0 + tl.arange(0, BLOCK_K)
        ic_m = ic_idx < IC
        # for each of the 9 taps, load x patch [BLOCK_M, BLOCK_K] and w [BLOCK_K, BLOCK_N], gemm
        for kh in tl.static_range(0, 3):
            ih = oh + kh
            for kw in tl.static_range(0, 3):
                iw = ow + kw
                # x offset: ((n*IC + ic)*H + ih)*W + iw
                x_off = (pid_n * IC + ic_idx[None, :]) * (H * W) + ih[:, None] * W + iw[:, None]
                x_mask = sp_mask[:, None] & ic_m[None, :]
                x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)
                # w offset: ((oc*IC + ic)*3 + kh)*3 + kw
                w_off = oc_offs[None, :] * (IC * 9) + ic_idx[:, None] * 9 + (kh * 3 + kw)
                w_mask = ic_m[:, None] & oc_mask[None, :]
                w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)
                acc += tl.dot(x_tile, w_tile)

    # add bias
    b = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b[None, :]

    # store: out[n, oc, oh, ow]
    out_off = (pid_n * OC + oc_offs[None, :]) * OHW + sp_offs[:, None]
    out_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


def conv3x3(x, w, b):
    N, IC, H, W = x.shape
    OC = w.shape[0]
    OH = H - 2
    OW = W - 2
    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (N, triton.cdiv(OC, BLOCK_N), triton.cdiv(OH * OW, BLOCK_M))
    conv3x3_kernel[grid](
        x, w, b, out,
        N, IC, H, W, OC, OH, OW,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=16, num_stages=3),
    ],
    key=['HW'],
)
@triton.jit
def instance_norm_div_kernel(
    x_ptr, out_ptr,
    HW,
    inv_div,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * HW

    sum_x = 0.0
    sum_x2 = 0.0
    for off in range(0, HW, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < HW
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    mean = sum_x / HW
    var = sum_x2 / HW - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    scale = rstd * inv_div
    shift = -mean * scale

    for off in range(0, HW, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < HW
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        y = x * scale + shift
        tl.store(out_ptr + base + idx, y, mask=mask)


def instance_norm_div(x: torch.Tensor, divide_by: float, eps: float = 1e-5):
    x = x.contiguous()
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)
    grid = (N * C,)
    instance_norm_div_kernel[grid](
        x, out,
        HW,
        1.0 / divide_by,
        eps,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.divide_by = float(divide_by)
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous()
        if self.kernel_size == 3 and x.is_cuda and x.dtype == torch.float32:
            w = self.conv.weight.contiguous()
            b = self.conv.bias.contiguous()
            x = conv3x3(x, w, b)
        else:
            x = self.conv(x)
        x = instance_norm_div(x, self.divide_by, eps=1e-5)
        return x