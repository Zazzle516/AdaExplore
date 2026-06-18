import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['N_OUT', 'OC', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv2d_div_lrelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, IC, H, W,
    OC,
    OH, OW,
    N_OUT,  # B * OH * OW
    inv_div,
    neg_slope,
    KH: tl.constexpr,
    KW: tl.constexpr,
    K_TOTAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_oc = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k = tl.arange(0, BLOCK_K)

    mask_n = offs_n < N_OUT
    mask_oc = offs_oc < OC

    # decode n -> (b, oh, ow)
    ow = offs_n % OW
    tmp = offs_n // OW
    oh = tmp % OH
    b = tmp // OH

    # base offset into x for each output point (with ic=kh=kw=0)
    x_n_base = (b * IC) * H * W + oh * W + ow  # [BLOCK_N]

    # Weight is contiguous [OC, IC*KH*KW]
    # We'll load w_tile [BLOCK_K, BLOCK_OC] by indexing w[offs_oc, k]
    acc = tl.zeros((BLOCK_N, BLOCK_OC), dtype=tl.float32)

    HW = H * W
    KHW = KH * KW

    for k_start in range(0, K_TOTAL, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]
        mask_k = k_idx < K_TOTAL

        # decode k -> (ic, kh, kw)
        kw_i = k_idx % KW
        tmp_k = k_idx // KW
        kh_i = tmp_k % KH
        ic_i = tmp_k // KH

        # x offsets: [BLOCK_N, BLOCK_K]
        x_k_off = ic_i * HW + kh_i * W + kw_i  # [BLOCK_K]
        x_offsets = x_n_base[:, None] + x_k_off[None, :]
        x_mask = mask_n[:, None] & mask_k[None, :]
        x_tile = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)  # [BLOCK_N, BLOCK_K]

        # w offsets: w[oc, k] = oc * K_TOTAL + k -> shape [BLOCK_K, BLOCK_OC]
        w_offsets = k_idx[:, None] + offs_oc[None, :] * K_TOTAL
        w_mask = mask_k[:, None] & mask_oc[None, :]
        w_tile = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_OC]

        acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    # bias
    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + bias[None, :]

    # divide
    acc = acc * inv_div

    # leaky relu
    acc = tl.where(acc >= 0, acc, acc * neg_slope)

    # store: output layout [B, OC, OH, OW]
    nhw_offset = b * (OC * OH * OW) + oh * OW + ow  # [BLOCK_N]
    out_offset = nhw_offset[:, None] + offs_oc[None, :] * (OH * OW)
    mask = mask_n[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_offset, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = float(divisor)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv.weight.contiguous().cuda()
        bias = self.conv.bias.contiguous().cuda()

        B, IC, H, W = x.shape
        OC, _, KH, KW = weight.shape
        OH = H - KH + 1
        OW = W - KW + 1

        out = torch.empty((B, OC, OH, OW), device=x.device, dtype=x.dtype)
        N_OUT = B * OH * OW
        inv_div = 1.0 / self.divisor
        neg_slope = 0.01

        K_TOTAL = IC * KH * KW

        grid = lambda meta: (
            triton.cdiv(OC, meta['BLOCK_OC']),
            triton.cdiv(N_OUT, meta['BLOCK_N']),
        )

        conv2d_div_lrelu_kernel[grid](
            x, weight, bias, out,
            B, IC, H, W,
            OC,
            OH, OW,
            N_OUT,
            inv_div,
            neg_slope,
            KH=KH, KW=KW, K_TOTAL=K_TOTAL,
        )
        return out