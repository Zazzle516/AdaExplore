import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_sub_rowsum_atomic_kernel(
    X_ptr, Wt_ptr, b_ptr, sub_ptr, RowSum_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wtk, stride_wtn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
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

    x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = Wt_ptr + (offs_k[:, None] * stride_wtk + offs_n[None, :] * stride_wtn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if EVEN_K:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            x = tl.load(x_ptrs)
            w = tl.load(w_ptrs)
            acc += tl.dot(x, w)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wtk
    else:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_remaining = K - k * BLOCK_K
            x_mask = offs_k[None, :] < k_remaining
            w_mask = offs_k[:, None] < k_remaining
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)
            w = tl.load(w_ptrs, mask=w_mask, other=0.0)
            acc += tl.dot(x, w)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wtk

    n_mask = offs_n < N
    b = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    s = tl.load(sub_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + b[None, :] - s[None, :]
    acc = tl.where(n_mask[None, :], acc, 0.0)

    row_partial = tl.sum(acc, axis=1)  # [BLOCK_M]

    m_mask = offs_m < M
    tl.atomic_add(RowSum_ptr + offs_m, row_partial, mask=m_mask)


@triton.jit
def gelu_residual_kernel(
    X_ptr, RowSum_ptr, OUT_ptr,
    M, K, N,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    rs = tl.load(RowSum_ptr + pid_m)
    s = rs / N  # mean; lse over single elem = identity
    inv_sqrt2 = 0.7071067811865475
    g = 0.5 * s * (1.0 + tl.math.erf(s * inv_sqrt2))

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs_k < K
    x = tl.load(X_ptr + pid_m * K + offs_k, mask=mask, other=0.0)
    out = x + g
    tl.store(OUT_ptr + pid_m * K + offs_k, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        self._Wt_cache = None
        self._Wt_version = -1
        self._Wt_device = None

    def _get_Wt(self):
        W = self.gemm.weight
        if (self._Wt_cache is None
                or self._Wt_version != W._version
                or self._Wt_device != W.device):
            self._Wt_cache = W.t().contiguous()
            self._Wt_version = W._version
            self._Wt_device = W.device
        return self._Wt_cache

    def forward(self, x):
        x = x.contiguous()
        M, K = x.shape
        N = self.out_features

        Wt = self._get_Wt()
        if self.gemm.bias is not None:
            b = self.gemm.bias.contiguous()
        else:
            b = torch.zeros(N, device=x.device, dtype=x.dtype)
        sub = self.subtract.contiguous()

        RowSum = torch.zeros((M,), device=x.device, dtype=torch.float32)

        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

        even_k = (K % 64 == 0)

        gemm_sub_rowsum_atomic_kernel[grid](
            x, Wt, b, sub, RowSum,
            M, N, K,
            x.stride(0), x.stride(1),
            Wt.stride(0), Wt.stride(1),
            EVEN_K=even_k,
        )

        out = torch.empty_like(x)
        BLOCK_K = 1024
        grid2 = (M, triton.cdiv(K, BLOCK_K))
        gelu_residual_kernel[grid2](x, RowSum, out, M, K, N, BLOCK_K=BLOCK_K)

        return out