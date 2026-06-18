import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'N_OUT_HW', 'IC_KH_KW'],
)
@triton.jit
def conv_relu_bias_kernel(
    x_ptr, w_ptr, b_ptr, bias_add_ptr, out_ptr,
    N, IC, H, W,
    OC, KH, KW,
    OH, OW,
    OUT_HW,        # OH*OW
    N_OUT_HW,      # N*OH*OW
    IC_KH_KW,      # IC*KH*KW
    KH_KW,         # KH*KW
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)              # OC tile
    pid_n = tl.program_id(1)              # N * OUT_HW tile

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)        # [BLOCK_M] -> oc
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)        # [BLOCK_N] -> flat (n,oh,ow)

    mask_m = offs_m < OC
    mask_n = offs_n < N_OUT_HW

    # decompose flat n-axis
    n_idx = offs_n // OUT_HW
    hw    = offs_n % OUT_HW
    oh    = hw // OW
    ow    = hw % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    # k layout: k = (kh*KW + kw) * IC + ic  -> ic innermost (contiguous)
    # weight is (OC, KH, KW, IC) contiguous  -> w_offset = oc*IC_KH_KW + k
    # input  is (N, H, W, IC) contiguous     -> x_offset = n*H*W*IC + (oh+kh)*W*IC + (ow+kw)*IC + ic

    H_W_IC = H * W * IC
    W_IC   = W * IC

    for k_start in range(0, IC_KH_KW, BLOCK_K):
        k = k_start + offs_k
        mask_k = k < IC_KH_KW

        khkw = k // IC
        ic   = k % IC
        kh   = khkw // KW
        kw   = khkw % KW

        # weight tile [BLOCK_M, BLOCK_K]
        w_offsets = offs_m[:, None] * IC_KH_KW + k[None, :]
        w_mask = mask_m[:, None] & mask_k[None, :]
        w_tile = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)

        # input tile [BLOCK_K, BLOCK_N]
        x_offsets = (n_idx[None, :] * H_W_IC
                     + (oh[None, :] + kh[:, None]) * W_IC
                     + (ow[None, :] + kw[:, None]) * IC
                     + ic[:, None])
        x_mask = mask_k[:, None] & mask_n[None, :]
        x_tile = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

        acc += tl.dot(w_tile, x_tile)

    # conv bias
    b = tl.load(b_ptr + offs_m, mask=mask_m, other=0.0)
    acc = acc + b[:, None]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # learned bias
    bias_add = tl.load(bias_add_ptr + offs_m, mask=mask_m, other=0.0)
    acc = acc + bias_add[:, None]

    # store: out is (N, OH, OW, OC) contiguous
    # out_offset = n * OH*OW*OC + oh * OW*OC + ow * OC + oc
    OH_OW_OC = OUT_HW * OC
    OW_OC    = OW * OC
    out_offsets = (n_idx[None, :] * OH_OW_OC
                   + oh[None, :] * OW_OC
                   + ow[None, :] * OC
                   + offs_m[:, None])
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0]
        OH = H - KH + 1
        OW = W - KW + 1

        # NHWC layout
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        # weight (OC, IC, KH, KW) -> (OC, KH, KW, IC)
        w_nhwc = self.conv.weight.permute(0, 2, 3, 1).contiguous()
        cb = self.conv.bias.contiguous()
        bias_add = self.bias.contiguous().view(-1)

        out_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

        OUT_HW = OH * OW
        N_OUT_HW = N * OUT_HW
        IC_KH_KW = IC * KH * KW
        KH_KW = KH * KW

        grid = lambda meta: (
            triton.cdiv(OC, meta['BLOCK_M']),
            triton.cdiv(N_OUT_HW, meta['BLOCK_N']),
        )

        conv_relu_bias_kernel[grid](
            x_nhwc, w_nhwc, cb, bias_add, out_nhwc,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW,
            OUT_HW, N_OUT_HW, IC_KH_KW, KH_KW,
        )
        # back to NCHW
        return out_nhwc.permute(0, 3, 1, 2).contiguous()