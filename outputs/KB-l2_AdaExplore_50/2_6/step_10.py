import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def softmax_pool_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    P: tl.constexpr,  # combined pool factor (4)
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

    d_start = od * P
    h_start = oh * P
    w_start = ow * P

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    # accumulator for max over pooling window
    max_vals = tl.full([BLOCK_C], -float('inf'), dtype=tl.float32)

    # iterate over P*P*P window
    for di in tl.static_range(P):
        for hi in tl.static_range(P):
            for wi in tl.static_range(P):
                d = d_start + di
                h = h_start + hi
                w = w_start + wi
                # base pointer for (n, :, d, h, w)
                base = n * C * D * H * W + d * H * W + h * W + w
                # load all C channels
                x_offs = base + c_offs * (D * H * W)
                vals = tl.load(x_ptr + x_offs, mask=c_mask, other=-float('inf'))

                # compute softmax over channels: need max and sum of exp
                m = tl.max(tl.where(c_mask, vals, -float('inf')), axis=0)
                ex = tl.exp(vals - m)
                ex = tl.where(c_mask, ex, 0.0)
                s = tl.sum(ex, axis=0)
                sm = ex / s

                max_vals = tl.maximum(max_vals, sm)

    # store: output shape (N, C, OD, OH, OW)
    out_base = n * C * OD * OH * OW + od * OH * OW + oh * OW + ow
    out_offs = out_base + c_offs * (OD * OH * OW)
    tl.store(out_ptr + out_offs, max_vals, mask=c_mask)


def softmax_pool_fused(x: torch.Tensor, pool_factor: int = 4):
    N, C, D, H, W = x.shape
    OD = D // pool_factor
    OH = H // pool_factor
    OW = W // pool_factor
    out = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)

    # next power of 2 for C
    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2
    BLOCK_C = max(BLOCK_C, 16)

    grid = (N * OD * OH * OW,)
    softmax_pool_kernel[grid](
        x, out,
        N, C, D, H, W,
        OD, OH, OW,
        P=pool_factor,
        BLOCK_C=BLOCK_C,
        num_warps=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self.pool_factor = pool_kernel_size * pool_kernel_size

    def forward(self, x):
        x = self.conv(x)
        # x: (N, C, D, H, W); softmax over C, then maxpool factor pool_factor
        # Ensure dimensions divisible
        N, C, D, H, W = x.shape
        pf = self.pool_factor
        if D % pf == 0 and H % pf == 0 and W % pf == 0:
            x = x.contiguous()
            return softmax_pool_fused(x, pf)
        else:
            x = torch.softmax(x, dim=1)
            x = F.max_pool3d(x, self.pool_kernel_size)
            x = F.max_pool3d(x, self.pool_kernel_size)
            return x