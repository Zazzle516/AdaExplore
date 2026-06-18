import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=2, num_stages=2),
    ],
    key=['N_OUT', 'OC', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv2d_div_lrelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, H, W,
    OC,
    OH, OW,
    N_OUT,
    neg_slope,
    IC: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    K_TOTAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, K_TOTAL)

    mask_m = offs_m < N_OUT
    mask_n = offs_n < OC

    # decode m -> (b, oh, ow). Output is NHWC layout: x is NHWC
    ow = offs_m % OW
    tmp = offs_m // OW
    oh = tmp % OH
    b = tmp // OH

    # x is NHWC: x[b, ih, iw, ic] = b*(H*W*IC) + ih*(W*IC) + iw*IC + ic
    # decode k -> (kh, kw, ic)
    ic_i = offs_k % IC
    tmp_k = offs_k // IC
    kw_i = tmp_k % KW
    kh_i = tmp_k // KW

    # input position
    ih = oh[:, None] + kh_i[None, :]   # [BLOCK_M, K_TOTAL]
    iw = ow[:, None] + kw_i[None, :]   # [BLOCK_M, K_TOTAL]
    x_off = b[:, None] * (H * W * IC) + ih * (W * IC) + iw * IC + ic_i[None, :]
    x_mask = mask_m[:, None]
    x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [BLOCK_M, K_TOTAL]

    # weight: [OC, K_TOTAL] in NHWC-friendly layout (OC, KH, KW, IC)
    # we need w_tile [K_TOTAL, BLOCK_N]
    w_off = offs_k[:, None] + offs_n[None, :] * K_TOTAL
    w_mask = mask_n[None, :]
    w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # [K_TOTAL, BLOCK_N]

    acc = tl.dot(x_tile, w_tile, allow_tf32=True)  # [BLOCK_M, BLOCK_N]

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]
    acc = tl.where(acc >= 0, acc, acc * neg_slope)

    # store: output layout [B, OC, OH, OW]
    nhw_offset = b * (OC * OH * OW) + oh * OW + ow  # [BLOCK_M]
    out_offset = nhw_offset[:, None] + offs_n[None, :] * (OH * OW)
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offset, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = float(divisor)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        with torch.no_grad():
            # Original weight: [OC, IC, KH, KW]. Reorder to [OC, KH, KW, IC] so
            # that the K-dimension (KH*KW*IC) matches NHWC traversal order.
            w = self.conv.weight / self.divisor  # [OC, IC, KH, KW]
            w = w.permute(0, 2, 3, 1).contiguous()  # [OC, KH, KW, IC]
            self.register_buffer('_scaled_weight', w)
            self.register_buffer('_scaled_bias', (self.conv.bias / self.divisor).contiguous())

    def forward(self, x):
        x = x.contiguous().cuda()
        # Permute input to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        weight = self._scaled_weight.to(x.device, non_blocking=True)
        bias = self._scaled_bias.to(x.device, non_blocking=True)

        B, IC, H, W = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1

        out = torch.empty((B, OC, OH, OW), device=x.device, dtype=x.dtype)
        N_OUT = B * OH * OW
        K_TOTAL = IC * KH * KW
        neg_slope = 0.01

        grid = lambda meta: (
            triton.cdiv(OC, meta['BLOCK_N']),
            triton.cdiv(N_OUT, meta['BLOCK_M']),
        )

        conv2d_div_lrelu_kernel[grid](
            x_nhwc, weight, bias, out,
            B, H, W,
            OC,
            OH, OW,
            N_OUT,
            neg_slope,
            IC=IC, KH=KH, KW=KW, K_TOTAL=K_TOTAL,
        )
        return out