import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# GEMM1: Y = sigmoid(X @ W1.T + b1)
# X: [M, K], W1_t: [K, N] (pre-transposed), b1: [N], Y: [M, N]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_sigmoid_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x = tl.load(x_ptrs, mask=offs_m[:, None] < M, other=0.0)
        w = tl.load(w_ptrs, mask=offs_n[None, :] < N, other=0.0)
        acc += tl.dot(x, w, allow_tf32=True)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    b = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]
    acc = 1.0 / (1.0 + tl.exp(-acc))

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# GEMM2 + LSE fused, single-pass numerically stable.
# Y: [M, H], W2_t: [H, O] (pre-transposed), b2: [O], out: [M]
# Each program handles BLOCK_M rows; the full output dim O is held in registers.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_H': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_H': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_H': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_H': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_H': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_H': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_H': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_H': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_H': 128}, num_warps=4, num_stages=2),
    ],
    key=['M', 'O', 'H'],
)
@triton.jit
def gemm_lse_kernel(
    Y_ptr, W_ptr, B_ptr, OUT_ptr,
    M, O, H,
    stride_ym, stride_yh,
    stride_wh, stride_wo,
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
    O_CONST: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    offs_o = tl.arange(0, O_CONST)
    offs_h = tl.arange(0, BLOCK_H)

    acc = tl.zeros((BLOCK_M, O_CONST), dtype=tl.float32)

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_h[None, :] * stride_yh
    w_ptrs = W_ptr + offs_h[:, None] * stride_wh + offs_o[None, :] * stride_wo

    for h in range(0, tl.cdiv(H, BLOCK_H)):
        y = tl.load(y_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs)
        acc += tl.dot(y, w, allow_tf32=True)
        y_ptrs += BLOCK_H * stride_yh
        w_ptrs += BLOCK_H * stride_wh

    b = tl.load(B_ptr + offs_o)
    acc = acc + b[None, :]

    row_max = tl.max(acc, axis=1)
    exp_acc = tl.exp(acc - row_max[:, None])
    row_sum = tl.sum(exp_acc, axis=1)
    out = row_max + tl.log(row_sum)
    tl.store(OUT_ptr + offs_m, out, mask=m_mask)


def gemm_sigmoid(x, w_t, b):
    M, K = x.shape
    N = w_t.shape[1]
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_sigmoid_kernel[grid](
        x, w_t, b, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        y.stride(0), y.stride(1),
    )
    return y


def gemm_lse(y, w_t, b):
    M, H = y.shape
    O = w_t.shape[1]
    out = torch.empty((M,), device=y.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
    gemm_lse_kernel[grid](
        y, w_t, b, out,
        M, O, H,
        y.stride(0), y.stride(1),
        w_t.stride(0), w_t.stride(1),
        O_CONST=O,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        # Pre-transpose weights once for contiguous fast-axis loads
        self.register_buffer('w1_t', self.linear1.weight.detach().t().contiguous().cuda())
        self.register_buffer('b1_', self.linear1.bias.detach().contiguous().cuda())
        self.register_buffer('w2_t', self.linear2.weight.detach().t().contiguous().cuda())
        self.register_buffer('b2_', self.linear2.bias.detach().contiguous().cuda())

    def forward(self, x):
        x = x.contiguous().cuda()
        y = gemm_sigmoid(x, self.w1_t, self.b1_)
        out = gemm_lse(y, self.w2_t, self.b2_)
        return out