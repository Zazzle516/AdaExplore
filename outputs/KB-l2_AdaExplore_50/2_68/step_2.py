import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _splitk_linear_min_sub_kernel(
    x_ptr, w_ptr, b_ptr, c_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # W is stored as (K, N) row-major after transpose, so stride_wk is along K, stride_wn=1
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    k_step = BLOCK_K * SPLIT_K
    # number of iterations this program does
    # iterate over k chunks belonging to this split
    k_iters = tl.cdiv(K - pid_k * BLOCK_K, k_step) if (pid_k * BLOCK_K) < K else 0

    for i in range(0, k_iters):
        cur_k = pid_k * BLOCK_K + i * k_step
        mask_k = (offs_k + i * k_step) < K
        x = tl.load(x_ptrs + i * k_step * stride_xk,
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs + i * k_step * stride_wk,
                    mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

    if SPLIT_K == 1:
        b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        acc += b[None, :]
        c = tl.load(c_ptr)
        acc = tl.minimum(acc, c) - c
        tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])
    else:
        tl.atomic_add(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _epilogue_kernel(
    out_ptr, b_ptr, c_ptr,
    M, N,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask = mask_m[:, None] & mask_n[None, :]
    val = tl.load(ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    val += b[None, :]
    c = tl.load(c_ptr)
    val = tl.minimum(val, c) - c
    tl.store(ptrs, val, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))
        self.in_features = in_features
        self.out_features = out_features
        # Pre-transpose weight: store as (K, N) contiguous so inner load along N is coalesced
        with torch.no_grad():
            wt = self.linear.weight.detach().t().contiguous()  # (K, N)
        self.register_buffer('weight_t', wt)

    def forward(self, x):
        x = x.contiguous()
        M, K = x.shape
        N = self.out_features
        W = self.weight_t  # (K, N), contiguous, stride_wk=N, stride_wn=1
        # Refresh weight_t in case linear.weight changed (not expected at inference)
        b = self.linear.bias

        out = torch.zeros((M, N), device=x.device, dtype=x.dtype)

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
            meta['SPLIT_K'],
        )
        _splitk_linear_min_sub_kernel[grid](
            x, W, b, self.constant, out,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            out.stride(0), out.stride(1),
        )

        # Epilogue: add bias + min - c (only when SPLIT_K > 1, kernel didn't fuse)
        # Determine if SPLIT_K was > 1 by checking the autotuner's chosen config
        best_cfg = _splitk_linear_min_sub_kernel.best_config
        if best_cfg.kwargs.get('SPLIT_K', 1) > 1:
            BM, BN = 32, 128
            grid2 = (triton.cdiv(M, BM), triton.cdiv(N, BN))
            _epilogue_kernel[grid2](
                out, b, self.constant,
                M, N,
                out.stride(0), out.stride(1),
                BLOCK_M=BM, BLOCK_N=BN,
            )
        return out