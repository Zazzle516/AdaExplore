import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['C_IN', 'C_OUT', 'H_OUT', 'W_OUT', 'KH', 'KW'],
)
@triton.jit
def conv2d_mish_gemm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_IN, H_IN, W_IN,
    C_OUT, H_OUT, W_OUT,
    KH: tl.constexpr, KW: tl.constexpr,
    SUB: tl.constexpr,
    BLOCK_M: tl.constexpr,  # output channels tile
    BLOCK_N: tl.constexpr,  # spatial (N*H_OUT*W_OUT) tile
    BLOCK_K: tl.constexpr,  # reduction tile
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    HW_OUT = H_OUT * W_OUT
    NHW_OUT = N * HW_OUT
    HW_IN = H_IN * W_IN
    KHW = KH * KW
    K_TOT = C_IN * KHW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # OC dim
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # spatial dim (over N*HW)

    mask_m = offs_m < C_OUT
    mask_n = offs_n < NHW_OUT

    # decode spatial index
    n_idx = offs_n // HW_OUT
    rem = offs_n % HW_OUT
    oh = rem // W_OUT
    ow = rem % W_OUT

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # iterate over reduction dim K = C_IN * KH * KW
    for k_start in range(0, K_TOT, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_TOT

        # decompose k -> (ic, kh, kw)
        ic = offs_k // KHW
        kk = offs_k % KHW
        kh = kk // KW
        kw = kk % KW

        # weight: shape [C_OUT, K_TOT] row-major
        w_off = offs_m[:, None] * K_TOT + offs_k[None, :]
        w_mask = mask_m[:, None] & mask_k[None, :]
        w = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

        # input: x[n, ic, oh+kh, ow+kw]
        x_off = (n_idx[None, :] * (C_IN * HW_IN)
                 + ic[:, None] * HW_IN
                 + (oh[None, :] + kh[:, None]) * W_IN
                 + (ow[None, :] + kw[:, None]))
        x_mask = mask_k[:, None] & mask_n[None, :]
        x = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

        acc += tl.dot(w, x, allow_tf32=True)

    # bias
    b = tl.load(b_ptr + offs_m, mask=mask_m, other=0.0)
    acc = acc + b[:, None]
    acc = acc - SUB

    # mish: x * tanh(softplus(x)), softplus stable
    ax = tl.abs(acc)
    sp = tl.log(1.0 + tl.exp(-ax)) + tl.maximum(acc, 0.0)
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    out = acc * th

    # store: out has shape [N, C_OUT, H_OUT, W_OUT]
    # offs_n decomposes to (n_idx, oh, ow)
    out_off = n_idx[None, :] * (C_OUT * HW_OUT) + offs_m[:, None] * HW_OUT + rem[None, :]
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_off, out, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, C_IN, H_IN, W_IN = x.shape
        KH = KW = self.kernel_size
        H_OUT = H_IN - KH + 1
        W_OUT = W_IN - KW + 1
        C_OUT = self.out_channels

        out = torch.empty((N, C_OUT, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

        SUB = float(self.subtract_value_1 + self.subtract_value_2)

        # weight is already [C_OUT, C_IN, KH, KW] contiguous -> view as [C_OUT, K_TOT]
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()

        NHW_OUT = N * H_OUT * W_OUT

        grid = lambda meta: (
            triton.cdiv(C_OUT, meta['BLOCK_M']),
            triton.cdiv(NHW_OUT, meta['BLOCK_N']),
        )

        conv2d_mish_gemm_kernel[grid](
            x, w, b, out,
            N, C_IN, H_IN, W_IN,
            C_OUT, H_OUT, W_OUT,
            KH, KW,
            SUB,
        )
        return out