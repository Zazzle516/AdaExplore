import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['OC', 'N_OUT_HW', 'IC'],
)
@triton.jit
def conv_relu_bias_kernel(
    x_ptr, w_ptr, b_ptr, bias_add_ptr, out_ptr,
    N, IC, H, W,
    OC,
    OH, OW,
    OUT_HW,        # OH*OW
    N_OUT_HW,      # N*OH*OW
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # M axis: output spatial (N * OH * OW)
    # N axis: OC
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # flat (n, oh, ow)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # oc

    mask_m = offs_m < N_OUT_HW
    mask_n = offs_n < OC

    n_idx = offs_m // OUT_HW
    hw    = offs_m % OUT_HW
    oh    = hw // OW
    ow    = hw % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    H_W_IC = H * W * IC
    W_IC   = W * IC
    IC_KH_KW = IC * KH * KW

    offs_k = tl.arange(0, BLOCK_K)

    # Outer loop over (kh, kw) — unrolled at compile time since KH, KW are constexpr
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            # base input offset for this kh, kw: input is NHWC
            # x[n, oh+kh, ow+kw, ic]
            base_x = (n_idx * H_W_IC
                      + (oh + kh) * W_IC
                      + (ow + kw) * IC)   # [BLOCK_M]
            # weight base: weight is (OC, KH, KW, IC), w_ptr at [oc, kh, kw, ic]
            base_w = offs_n * IC_KH_KW + (kh * KW + kw) * IC  # [BLOCK_N]

            for ic_start in range(0, IC, BLOCK_K):
                ic = ic_start + offs_k

                # x_tile [BLOCK_M, BLOCK_K] — IC is a multiple of BLOCK_K, so no ic mask
                x_off = base_x[:, None] + ic[None, :]
                x_tile = tl.load(x_ptr + x_off, mask=mask_m[:, None], other=0.0)

                # w_tile [BLOCK_K, BLOCK_N]
                w_off = ic[:, None] + base_w[None, :]
                w_tile = tl.load(w_ptr + w_off, mask=mask_n[None, :], other=0.0)

                acc += tl.dot(x_tile, w_tile)

    # conv bias
    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # learned bias
    bias_add = tl.load(bias_add_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias_add[None, :]

    # store: out is NHWC: (N, OH, OW, OC)
    OH_OW_OC = OUT_HW * OC
    OW_OC = OW * OC
    out_off = (n_idx[:, None] * OH_OW_OC
               + oh[:, None] * OW_OC
               + ow[:, None] * OC
               + offs_n[None, :])
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]

    def forward(self, x):
        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
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

        grid = lambda meta: (
            triton.cdiv(N_OUT_HW, meta['BLOCK_M']),
            triton.cdiv(OC, meta['BLOCK_N']),
        )

        conv_relu_bias_kernel[grid](
            x_nhwc, w_nhwc, cb, bias_add, out_nhwc,
            N, IC, H, W,
            OC,
            OH, OW,
            OUT_HW, N_OUT_HW,
            KH, KW,
        )

        return out_nhwc.permute(0, 3, 1, 2).contiguous()