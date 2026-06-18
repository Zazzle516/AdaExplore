import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_pool_sigmoid_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, KH, KW,
    OH, OW,  # conv output dims
    PH, PW,  # pooled dims
    POOL: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, oc_tile, pool_tile)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_p = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)

    oc_mask = oc_offs < OC
    p_mask = p_offs < (PH * PW)

    ph = p_offs // PW
    pw = p_offs % PW

    # pooled output = sum over POOL x POOL window of conv output / (POOL*POOL), then sigmoid
    # conv output at (oh, ow) = sum_{ic, kh, kw} x[n, ic, oh+kh, ow+kw] * w[oc, ic, kh, kw] + b[oc]
    # accumulator: [BLOCK_OC, BLOCK_P]
    acc = tl.zeros((BLOCK_OC, BLOCK_P), dtype=tl.float32)

    # Iterate pool window
    for dy in tl.static_range(0, POOL):
        for dx in tl.static_range(0, POOL):
            oh = ph * POOL + dy  # [BLOCK_P]
            ow = pw * POOL + dx  # [BLOCK_P]
            # Compute conv at (oh, ow) for all oc in tile
            # sum over ic, kh, kw
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    ih = oh + kh
                    iw = ow + kw
                    for ic in range(0, IC):
                        # load x[n, ic, ih, iw] for all p
                        x_idx = pid_n * IC * H * W + ic * H * W + ih * W + iw
                        x_val = tl.load(x_ptr + x_idx, mask=p_mask, other=0.0)  # [BLOCK_P]
                        # load w[oc, ic, kh, kw] for all oc
                        w_idx = oc_offs * IC * KH * KW + ic * KH * KW + kh * KW + kw
                        w_val = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                        acc += w_val[:, None] * x_val[None, :]

    # add bias (POOL*POOL times since summed over pool window)
    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b_val[:, None] * (POOL * POOL)

    # divide by pool area
    acc = acc / (POOL * POOL)

    # sigmoid
    acc = 1.0 / (1.0 + tl.exp(-acc))

    # mask out invalid positions
    valid = oc_mask[:, None] & p_mask[None, :]
    acc = tl.where(valid, acc, 0.0)

    # sum reduction over oc and p
    partial = tl.sum(acc)

    # atomic add to output[n]
    tl.atomic_add(out_ptr + pid_n, partial)


def conv_pool_sigmoid_sum(x, weight, bias, pool_size):
    N, IC, H, W = x.shape
    OC, _, KH, KW = weight.shape
    OH = H - KH + 1
    OW = W - KW + 1
    PH = OH // pool_size
    PW = OW // pool_size

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    out = torch.zeros(N, dtype=torch.float32, device=x.device)

    BLOCK_P = 64
    BLOCK_OC = 16

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(PH * PW, BLOCK_P))

    conv_pool_sigmoid_sum_kernel[grid](
        x, weight, bias, out,
        N, IC, H, W,
        OC, KH, KW,
        OH, OW,
        PH, PW,
        POOL=pool_size,
        BLOCK_P=BLOCK_P,
        BLOCK_OC=BLOCK_OC,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        return conv_pool_sigmoid_sum(x, self.conv.weight, self.conv.bias, self.pool_kernel_size)