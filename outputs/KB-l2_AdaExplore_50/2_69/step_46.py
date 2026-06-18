import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=2, num_stages=2),
    ],
    key=['N', 'OC', 'IC', 'H', 'W', 'KH', 'KW'],
)
@triton.jit
def conv_hardswish_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, H, W,
    OH, OW,
    IC: tl.constexpr, OC: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    K: tl.constexpr,  # IC*KH*KW
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # output spatial-batch index
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output channel index

    NHW = N * OH * OW
    mask_m = offs_m < NHW
    mask_n = offs_n < OC

    # decode n, oh, ow from offs_m
    n_idx = offs_m // (OH * OW)
    rem = offs_m % (OH * OW)
    oh_idx = rem // OW
    ow_idx = rem % OW

    # K = IC*KH*KW, index k -> (kh, kw, ic) for NHWC layout
    offs_k = tl.arange(0, K)
    kh_idx = offs_k // (KW * IC)
    kw_idx = (offs_k // IC) % KW
    ic_idx = offs_k % IC

    # x is NHWC contiguous: stride (H*W*IC, W*IC, IC, 1)
    # input row m loads at (n, oh+kh, ow+kw, ic) for k in [0,K)
    # base for row m: n*H*W*IC + oh*W*IC + ow*IC
    x_row_base = n_idx * (H * W * IC) + oh_idx * (W * IC) + ow_idx * IC  # [BLOCK_M]
    # offset per k: kh*W*IC + kw*IC + ic
    x_k_off = kh_idx * (W * IC) + kw_idx * IC + ic_idx  # [K]

    x_offs = x_row_base[:, None] + x_k_off[None, :]  # [BLOCK_M, K]
    x_vals = tl.load(x_ptr + x_offs, mask=mask_m[:, None], other=0.0)

    # weight reshape: (OC, K) where K layout is (kh, kw, ic) - matches above
    w_offs = offs_n[:, None] * K + offs_k[None, :]  # [BLOCK_N, K]
    w_vals = tl.load(w_ptr + w_offs, mask=mask_n[:, None], other=0.0)

    # acc = x @ w^T -> [BLOCK_M, BLOCK_N]
    acc = tl.dot(x_vals, tl.trans(w_vals))

    # bias
    b_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b_vals[None, :]

    # relu(hardswish(x))
    out = tl.where(acc > 0.0, acc * tl.minimum(acc + 3.0, 6.0) * (1.0 / 6.0), 0.0)

    # output is NHWC: (N, OH, OW, OC), stride (OH*OW*OC, OW*OC, OC, 1)
    out_row_base = n_idx * (OH * OW * OC) + oh_idx * (OW * OC) + ow_idx * OC  # [BLOCK_M]
    out_offs = out_row_base[:, None] + offs_n[None, :]  # [BLOCK_M, BLOCK_N]
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-arrange weight in (OC, KH, KW, IC) -> (OC, K) layout for NHWC convolution
        with torch.no_grad():
            w = self.conv.weight.detach()  # (OC, IC, KH, KW)
            OC, IC, KH, KW = w.shape
            # we want w[oc, kh, kw, ic] -> permute to (OC, KH, KW, IC)
            w_nhwc = w.permute(0, 2, 3, 1).contiguous().view(OC, KH * KW * IC)
        self.register_buffer('weight_packed', w_nhwc.cuda(), persistent=False)
        self.register_buffer('bias_packed', self.conv.bias.detach().contiguous().cuda(), persistent=False)

    def forward(self, x):
        x = x.cuda()
        N, IC, H, W = x.shape
        KH = self.kernel_size
        KW = self.kernel_size
        OC = self.out_channels
        OH = H - KH + 1
        OW = W - KW + 1
        K = IC * KH * KW

        # Convert input to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        # Output in NHWC, then permute back at end
        out_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

        grid = lambda meta: (
            triton.cdiv(N * OH * OW, meta['BLOCK_M']),
            triton.cdiv(OC, meta['BLOCK_N']),
        )

        conv_hardswish_relu_kernel[grid](
            x_nhwc, self.weight_packed, self.bias_packed, out_nhwc,
            N, H, W,
            OH, OW,
            IC, OC,
            KH, KW,
            K,
        )

        return out_nhwc.permute(0, 3, 1, 2).contiguous()