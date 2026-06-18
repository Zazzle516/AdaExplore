import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _conv2d_double_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr,  # OC tile
    BLOCK_N: tl.constexpr,  # spatial tile
    BLOCK_K: tl.constexpr,  # IC tile
):
    pid_n = tl.program_id(0)        # batch index
    pid_m = tl.program_id(1)        # OC tile index
    pid_s = tl.program_id(2)        # spatial tile index

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)        # OC indices
    offs_s = pid_s * BLOCK_N + tl.arange(0, BLOCK_N)        # spatial indices
    oh = offs_s // OW
    ow = offs_s % OW

    mask_m = offs_m < OC
    mask_s = offs_s < (OH * OW)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # x layout: (N, IC, H, W) contiguous
    # w layout: (OC, IC, KH, KW) contiguous
    x_batch_ptr = x_ptr + pid_n * IC * H * W

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh   # stride=1, padding=0
            iw = ow + kw
            # for each ic block
            for ic_start in range(0, IC, BLOCK_K):
                offs_k = ic_start + tl.arange(0, BLOCK_K)
                mask_k = offs_k < IC

                # Load x: shape (BLOCK_K, BLOCK_N)
                x_offsets = (offs_k[:, None] * H * W) + (ih[None, :] * W) + iw[None, :]
                x_mask = mask_k[:, None] & mask_s[None, :]
                x_vals = tl.load(x_batch_ptr + x_offsets, mask=x_mask, other=0.0)

                # Load w: shape (BLOCK_M, BLOCK_K)
                w_offsets = (offs_m[:, None] * IC * KH * KW) + (offs_k[None, :] * KH * KW) + (kh * KW + kw)
                w_mask = mask_m[:, None] & mask_k[None, :]
                w_vals = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)

                acc += tl.dot(w_vals, x_vals)

    # Bias
    bias = tl.load(b_ptr + offs_m, mask=mask_m, other=0.0)
    acc = acc + bias[:, None]

    # First mish
    sp1 = tl.log(1.0 + tl.exp(acc))
    t1 = 2.0 * tl.sigmoid(2.0 * sp1) - 1.0
    y = acc * t1
    # Second mish
    sp2 = tl.log(1.0 + tl.exp(y))
    t2 = 2.0 * tl.sigmoid(2.0 * sp2) - 1.0
    z = y * t2

    # Store: out shape (N, OC, OH, OW)
    out_batch_ptr = out_ptr + pid_n * OC * OH * OW
    out_offsets = offs_m[:, None] * (OH * OW) + offs_s[None, :]
    out_mask = mask_m[:, None] & mask_s[None, :]
    tl.store(out_batch_ptr + out_offsets, z, mask=out_mask)


def conv2d_double_mish(x, w, b):
    N, IC, H, W = x.shape
    OC, _, KH, KW = w.shape
    OH = H - KH + 1
    OW = W - KW + 1

    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()
    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64

    grid = (N, triton.cdiv(OC, BLOCK_M), triton.cdiv(OH * OW, BLOCK_N))

    _conv2d_double_mish_kernel[grid](
        x, w, b, out,
        N, IC, H, W,
        OC, OH, OW,
        KH, KW,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return conv2d_double_mish(x, self.conv.weight, self.conv.bias)