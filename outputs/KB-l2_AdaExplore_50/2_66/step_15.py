import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
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

    # x is (M, K) row-major: K is contiguous on inner load
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    # w is pre-transposed to (K, N) row-major: N is contiguous on inner load
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Peeled K-loop: full tiles unmasked, tail masked
    k_full = (K // BLOCK_K) * BLOCK_K
    for k in range(0, k_full, BLOCK_K):
        x = tl.load(x_ptrs)
        w = tl.load(w_ptrs)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    if k_full < K:
        k_mask = offs_k < (K - k_full)
        x = tl.load(x_ptrs, mask=k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(x, w)

    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += b[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask)


@triton.jit
def softmax_kernel(
    inp_ptr, out_ptr, n_cols,
    stride_im, stride_om,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    inp_row = inp_ptr + row * stride_im
    out_row = out_ptr + row * stride_om

    # Pass 1: online max + sum (rescale)
    max_val = -float('inf')
    sum_exp = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        vals = tl.load(inp_row + offs, mask=mask, other=-float('inf'))
        block_max = tl.max(vals, axis=0)
        new_max = tl.maximum(max_val, block_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max)
        ex = tl.exp(vals - new_max)
        ex = tl.where(mask, ex, 0.0)
        sum_exp += tl.sum(ex, axis=0)
        max_val = new_max

    inv = 1.0 / sum_exp

    # Pass 2: write normalized output
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        vals = tl.load(inp_row + offs, mask=mask, other=-float('inf'))
        ex = tl.exp(vals - max_val) * inv
        tl.store(out_row + offs, ex, mask=mask)


def triton_linear(x, weight_t, bias):
    # weight_t is (K, N) row-major
    M, K = x.shape
    K2, N = weight_t.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    linear_kernel[grid](
        x, weight_t, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


def triton_softmax(x):
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK = 2048
    grid = (M,)
    softmax_kernel[grid](
        x, out, N,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK,
        num_warps=8,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)
        self.in_features = in_features
        self.out_features = out_features
        self._weight_t_cache = None

    def _get_weight_t(self):
        # Pre-transpose to (K, N) contiguous so N is the inner contiguous dim
        w = self.matmul.weight
        if (self._weight_t_cache is None
                or self._weight_t_cache.device != w.device
                or self._weight_t_cache.data_ptr() == 0):
            self._weight_t_cache = w.detach().t().contiguous().cuda()
        return self._weight_t_cache

    def forward(self, x):
        x = x.contiguous().cuda()
        w_t = self._get_weight_t()
        b = self.matmul.bias.contiguous().cuda()
        y = triton_linear(x, w_t, b)
        if self.training:
            y = self.dropout(y)
        out = triton_softmax(y)
        return out