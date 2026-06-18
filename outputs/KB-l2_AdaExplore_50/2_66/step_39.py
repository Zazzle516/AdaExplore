import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
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
):
    pid_n = tl.program_id(0)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs)
        w = tl.load(w_ptrs)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    b = tl.load(b_ptr + offs_n)
    acc += b[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc)


@triton.jit
def softmax_kernel(
    inp_ptr, out_ptr, n_cols,
    stride_im, stride_om,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    inp_row = inp_ptr + row * stride_im
    out_row = out_ptr + row * stride_om

    max_val = -float('inf')
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        vals = tl.load(inp_row + offs, mask=mask, other=-float('inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    sum_exp = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        vals = tl.load(inp_row + offs, mask=mask, other=-float('inf'))
        ex = tl.exp(vals - max_val)
        ex = tl.where(mask, ex, 0.0)
        sum_exp += tl.sum(ex, axis=0)

    inv = 1.0 / sum_exp

    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        vals = tl.load(inp_row + offs, mask=mask, other=-float('inf'))
        ex = tl.exp(vals - max_val) * inv
        tl.store(out_row + offs, ex, mask=mask)


def triton_linear(x, weight_t, bias):
    """
    x: (M, K)
    weight_t: (K, N)  -- pre-transposed weight
    bias: (N,)
    """
    M, K = x.shape
    K2, N = weight_t.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_N']),)
    linear_kernel[grid](
        x, weight_t, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=M,
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
        self._weight_t_cache = None

    def _get_weight_t(self):
        w = self.matmul.weight
        if (self._weight_t_cache is None
                or self._weight_t_cache.device != w.device
                or self._weight_t_cache.dtype != w.dtype
                or self._weight_t_cache.shape != (w.shape[1], w.shape[0])
                or self._weight_t_cache._version != getattr(self, '_weight_version', -1)):
            wt = w.t().contiguous().cuda()
            self._weight_t_cache = wt
            self._weight_version = w._version
        return self._weight_t_cache

    def forward(self, x):
        x = x.contiguous().cuda()
        wt = self._get_weight_t()
        b = self.matmul.bias.contiguous().cuda()
        y = triton_linear(x, wt, b)
        if self.training:
            y = self.dropout(y)
        out = triton_softmax(y)
        return out