import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OUT_HW', 'IC_KHW'],
)
@triton.jit
def conv_scale_min_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, KH, KW,
    OH, OW,
    scale,
    OUT_HW, IC_KHW,
    BLOCK_M: tl.constexpr,  # output spatial tile (along OH*OW)
    BLOCK_N: tl.constexpr,  # OC tile
    BLOCK_K: tl.constexpr,  # IC*KH*KW reduction tile
):
    pid_n = tl.program_id(0)  # batch
    pid_m = tl.program_id(1)  # output spatial tile
    pid_oc = tl.program_id(2)  # OC tile

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # output spatial idx
    offs_n = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)  # oc idx
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < OUT_HW
    mask_n = offs_n < OC

    oh = offs_m // OW
    ow = offs_m % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    KHW = KH * KW

    for k0 in range(0, IC_KHW, BLOCK_K):
        k_idx = k0 + offs_k  # [BLOCK_K]
        k_mask = k_idx < IC_KHW

        ic = k_idx // KHW
        rem = k_idx % KHW
        kh = rem // KW
        kw = rem % KW

        # Load x: [BLOCK_M, BLOCK_K]
        h_in = oh[:, None] + kh[None, :]
        w_in = ow[:, None] + kw[None, :]
        ic_b = ic[None, :]

        x_offset = (pid_n * IC * H * W
                    + ic_b * (H * W)
                    + h_in * W
                    + w_in)
        x_mask = mask_m[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)

        # Load w: [BLOCK_K, BLOCK_N]
        # w shape: (OC, IC, KH, KW) -> idx = oc*IC*KH*KW + ic*KH*KW + kh*KW + kw
        w_offset = (offs_n[None, :] * IC_KHW + k_idx[:, None])
        w_mask = k_mask[:, None] & mask_n[None, :]
        w_vals = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    # add bias
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]
    acc = acc * scale

    # mask out invalid OC slots so they don't participate in min
    INF = float('inf')
    acc = tl.where(mask_n[None, :], acc, INF)

    # min reduction along N (OC tile) dimension - this is partial min for this tile of OC
    partial_min = tl.min(acc, axis=1)  # [BLOCK_M]

    # Atomic min into output (one entry per (n, m))
    out_offset = pid_n * OUT_HW + offs_m
    tl.atomic_min(out_ptr + out_offset, partial_min, mask=mask_m)


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

        # Initialize output to +inf for atomic_min
        out = torch.full((N, 1, OH, OW), float('inf'), device=x.device, dtype=torch.float32)

        grid = lambda meta: (
            N,
            triton.cdiv(OUT_HW, meta['BLOCK_M']),
            triton.cdiv(OC, meta['BLOCK_N']),
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