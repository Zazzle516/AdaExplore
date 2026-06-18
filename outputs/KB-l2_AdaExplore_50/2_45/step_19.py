import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# GEMM1: Y = sigmoid(X @ W1.T + b1)
# X: [M, K], W1: [N, K] (so W1.T is [K, N]), b1: [N], Y: [M, N]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_sigmoid_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remain = K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remain), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < k_remain) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    b = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]
    # sigmoid
    acc = 1.0 / (1.0 + tl.exp(-acc))

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# GEMM2 + LogSumExp fused: out[m] = logsumexp(Y @ W2.T + b2)
# Y: [M, H], W2: [O, H], b2: [O], out: [M]
# Strategy: each program computes one row, iterates over output dim in tiles,
# computes the GEMM tile, applies bias, and does online logsumexp.
# But this requires the whole hidden dim per output tile. Use BLOCK_H to chunk K.
# To do online LSE we need full output for each row. Compute output row in chunks of BLOCK_O,
# and for each chunk compute full GEMM along H. Then merge into running (max, sum_exp).

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_O': 128, 'BLOCK_H': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_O': 128, 'BLOCK_H': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_O': 64, 'BLOCK_H': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_O': 128, 'BLOCK_H': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_O': 256, 'BLOCK_H': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_O': 64, 'BLOCK_H': 32}, num_warps=4, num_stages=4),
    ],
    key=['M', 'O', 'H'],
)
@triton.jit
def gemm_lse_kernel(
    Y_ptr, W_ptr, B_ptr, OUT_ptr,
    M, O, H,
    stride_ym, stride_yh,
    stride_wo, stride_wh,
    BLOCK_M: tl.constexpr, BLOCK_O: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    # Running stats per row
    running_max = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    num_o_blocks = tl.cdiv(O, BLOCK_O)
    for o_block in range(0, num_o_blocks):
        offs_o = o_block * BLOCK_O + tl.arange(0, BLOCK_O)
        o_mask = offs_o < O

        # Compute Z[BLOCK_M, BLOCK_O] = Y[BLOCK_M, H] @ W[BLOCK_O, H].T + b[BLOCK_O]
        acc = tl.zeros((BLOCK_M, BLOCK_O), dtype=tl.float32)

        offs_h = tl.arange(0, BLOCK_H)
        y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_h[None, :] * stride_yh
        w_ptrs = W_ptr + offs_o[None, :] * stride_wo + offs_h[:, None] * stride_wh

        for h in range(0, tl.cdiv(H, BLOCK_H)):
            h_remain = H - h * BLOCK_H
            y = tl.load(y_ptrs, mask=m_mask[:, None] & (offs_h[None, :] < h_remain), other=0.0)
            w = tl.load(w_ptrs, mask=(offs_h[:, None] < h_remain) & o_mask[None, :], other=0.0)
            acc += tl.dot(y, w)
            y_ptrs += BLOCK_H * stride_yh
            w_ptrs += BLOCK_H * stride_wh

        b = tl.load(B_ptr + offs_o, mask=o_mask, other=0.0)
        acc = acc + b[None, :]
        # Mask invalid columns to -inf so they don't contribute
        acc = tl.where(o_mask[None, :], acc, -float('inf'))

        # Online LSE merge
        block_max = tl.max(acc, axis=1)  # [BLOCK_M]
        new_max = tl.maximum(running_max, block_max)
        # Exponents
        exp_acc = tl.exp(acc - new_max[:, None])
        block_sum = tl.sum(exp_acc, axis=1)
        scale = tl.exp(running_max - new_max)
        running_sum = running_sum * scale + block_sum
        running_max = new_max

    out = running_max + tl.log(running_sum)
    tl.store(OUT_ptr + offs_m, out, mask=m_mask)


def gemm_sigmoid(x, w, b):
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    gemm_sigmoid_kernel[grid](
        x, w, b, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
    )
    return y


def gemm_lse(y, w, b):
    M, H = y.shape
    O = w.shape[0]
    out = torch.empty((M,), device=y.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
    gemm_lse_kernel[grid](
        y, w, b, out,
        M, O, H,
        y.stride(0), y.stride(1),
        w.stride(0), w.stride(1),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.contiguous().cuda()
        w1 = self.linear1.weight.contiguous()
        b1 = self.linear1.bias.contiguous()
        w2 = self.linear2.weight.contiguous()
        b2 = self.linear2.bias.contiguous()

        y = gemm_sigmoid(x, w1, b1)
        out = gemm_lse(y, w2, b2)
        return out