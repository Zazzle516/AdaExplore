import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OUT_HW', 'IC_KHW'],
)
@triton.jit
def conv_scale_min_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    OH, OW,
    scale,
    OUT_HW, IC_KHW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_oc = tl.arange(0, OC)
    mask_m = offs_m < OUT_HW

    oh = offs_m // OW
    ow = offs_m % OW

    KHW = KH * KW

    acc = tl.zeros((BLOCK_M, OC), dtype=tl.float32)

    x_batch_off = pid_n * IC * H * W

    for k_start in range(0, IC_KHW, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < IC_KHW

        ic = offs_k // KHW
        rem = offs_k % KHW
        kh = rem // KW
        kw = rem % KW

        # Weight tile [BLOCK_K, OC]
        w_offset = offs_oc[None, :] * IC_KHW + offs_k[:, None]
        w_vals = tl.load(w_ptr + w_offset, mask=mask_k[:, None], other=0.0)

        # X tile [BLOCK_M, BLOCK_K]
        h_in = oh[:, None] + kh[None, :]
        w_in = ow[:, None] + kw[None, :]
        x_offset = (x_batch_off
                    + ic[None, :] * (H * W)
                    + h_in * W
                    + w_in)
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_vals = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    bias = tl.load(b_ptr + offs_oc)
    acc = acc + bias[None, :]
    acc = acc * scale

    # min along OC
    min_val = tl.min(acc, axis=1)

    out_offset = pid_n * OUT_HW + offs_m
    tl.store(out_ptr + out_offset, min_val, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, H, W = x.shape
        OC, _, KH, KW = w.shape
        OH = H - KH + 1
        OW = W - KW + 1
        OUT_HW = OH * OW
        IC_KHW = IC * KH * KW

        out = torch.empty((N, 1, OH, OW), device=x.device, dtype=torch.float32)

        grid = lambda meta: (
            N,
            triton.cdiv(OUT_HW, meta['BLOCK_M']),
        )

        conv_scale_min_kernel[grid](
            x, w, b, out,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW,
            float(self.scale_factor),
            OUT_HW, IC_KHW,
        )

        return out