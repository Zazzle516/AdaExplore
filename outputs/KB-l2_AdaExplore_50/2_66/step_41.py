import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K', 'SPLIT_K'],
)
@triton.jit
def linear_splitk_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

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
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_step = BLOCK_K * SPLIT_K
    K_iters = tl.cdiv(K - pid_k * BLOCK_K, k_step)

    for i in range(0, K_iters):
        x = tl.load(x_ptrs)
        w = tl.load(w_ptrs)
        acc += tl.dot(x, w)
        x_ptrs += k_step * stride_xk
        w_ptrs += k_step * stride_wk

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    if SPLIT_K == 1:
        tl.store(out_ptrs, acc)
    else:
        tl.atomic_add(out_ptrs, acc)


@triton.jit
def bias_softmax_kernel(
    inp_ptr, b_ptr, out_ptr, n_cols,
    stride_im, stride_om,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    inp_row = inp_ptr + row * stride_im
    out_row = out_ptr + row * stride_om

    # First pass: max (with bias added)
    max_val = -float('inf')
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        vals = tl.load(inp_row + offs, mask=mask, other=-float('inf'))
        bias = tl.load(b_ptr + offs, mask=mask, other=0.0)
        vals = vals + bias
        vals = tl.where(mask, vals, -float('inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Second pass: sum exp
    sum_exp = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        vals = tl.load(inp_row + offs, mask=mask, other=0.0)
        bias = tl.load(b_ptr + offs, mask=mask, other=0.0)
        vals = vals + bias
        ex = tl.exp(vals - max_val)
        ex = tl.where(mask, ex, 0.0)
        sum_exp += tl.sum(ex, axis=0)

    inv = 1.0 / sum_exp

    # Third pass: write
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        vals = tl.load(inp_row + offs, mask=mask, other=0.0)
        bias = tl.load(b_ptr + offs, mask=mask, other=0.0)
        vals = vals + bias
        ex = tl.exp(vals - max_val) * inv
        tl.store(out_row + offs, ex, mask=mask)


def triton_linear(x, weight, M, N, K, split_k=4):
    if split_k == 1:
        out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    else:
        out = torch.zeros((M, N), device=x.device, dtype=torch.float32)

    GROUP_M = 8
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
        split_k,
    )
    linear_splitk_kernel[grid](
        x, weight, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        SPLIT_K=split_k,
        GROUP_M=GROUP_M,
    )
    return out


def triton_bias_softmax(x, bias):
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK = 4096
    grid = (M,)
    bias_softmax_kernel[grid](
        x, bias, out, N,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK,
        num_warps=16,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.matmul.weight.contiguous().cuda()
        b = self.matmul.bias.contiguous().cuda()
        M, K = x.shape
        N = w.shape[0]

        # split-K GEMM (without bias)
        y = triton_linear(x, w, M, N, K, split_k=4)

        if self.training and self.dropout_p > 0:
            # Apply bias first, then dropout, then softmax via standard path
            y = y + b
            y = self.dropout(y)
            return torch.softmax(y, dim=1)

        # Fused bias + softmax
        out = triton_bias_softmax(y, b)
        return out