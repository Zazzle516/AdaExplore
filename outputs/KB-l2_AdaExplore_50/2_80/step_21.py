import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_rowmax_kernel(
    x_ptr, w_ptr, b_ptr, partial_ptr,
    M, N, K, NUM_PID_N,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_pn, stride_pm,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # GROUP_M swizzle
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # Load W as [BLOCK_K, BLOCK_N] directly to avoid tl.trans in the hot loop
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        x_mask = mask_m[:, None] & (offs_k[None, :] < k_remaining)
        w_mask = (offs_k[:, None] < k_remaining) & mask_n[None, :]
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # add bias
    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]

    # mask invalid lanes to -inf for max
    acc = tl.where(mask_n[None, :], acc, -float('inf'))

    # row max within tile
    row_max = tl.max(acc, axis=1)  # [BLOCK_M]

    # write partial max to partial[pid_n, offs_m]
    p_ptrs = partial_ptr + pid_n * stride_pn + offs_m * stride_pm
    tl.store(p_ptrs, row_max, mask=mask_m)


@triton.jit
def reduce_max_gelu_kernel(
    partial_ptr, out_ptr,
    M, NUM_PID_N,
    stride_pn, stride_pm,
    BLOCK_M: tl.constexpr, BLOCK_PN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    offs_pn = tl.arange(0, BLOCK_PN)
    mask_pn = offs_pn < NUM_PID_N

    p_ptrs = partial_ptr + offs_pn[:, None] * stride_pn + offs_m[None, :] * stride_pm
    mask = mask_pn[:, None] & mask_m[None, :]
    vals = tl.load(p_ptrs, mask=mask, other=-float('inf'))
    row_max = tl.max(vals, axis=0)  # [BLOCK_M]

    # row_max has shape (M,) representing (M,1) reduced. mean over dim=1 of (M,1) is itself.
    # diff = row_max - row_max = 0; gelu(0) = 0. We compute it explicitly.
    mean = row_max  # mean over single-element axis
    diff = row_max - mean
    inv_sqrt2 = 0.70710678118654752440
    y = 0.5 * diff * (1.0 + tl.erf(diff * inv_sqrt2))
    tl.store(out_ptr + offs_m, y, mask=mask_m)


@triton.jit
def gelu_kernel(
    inp_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    # After max-keepdim along dim=1 yields shape (B,1), then x - mean over dim=1 of a (B,1) tensor is 0.
    # So input to gelu is 0 -> output is 0. But we still compute properly for correctness.
    # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.70710678118654752440
    y = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    tl.store(out_ptr + offs, y, mask=mask)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def gemm_rowmax_gelu(x, weight, bias):
    M, K = x.shape
    N, Kw = weight.shape
    assert K == Kw

    # We need to know num_pid_n to allocate partial buffer; but BLOCK_N is autotuned.
    # Use a worst-case allocation based on smallest BLOCK_N in configs (128).
    MAX_NUM_PID_N = triton.cdiv(N, 128)  # smallest BLOCK_N is 128

    partial = torch.empty((MAX_NUM_PID_N, M), device=x.device, dtype=torch.float32)
    out = torch.empty((M,), device=x.device, dtype=torch.float32)

    def grid(meta):
        num_pid_m = triton.cdiv(M, meta['BLOCK_M'])
        num_pid_n = triton.cdiv(N, meta['BLOCK_N'])
        return (num_pid_m * num_pid_n,)

    # Need to pass NUM_PID_N actually used; but BLOCK_N is in meta. We pass via meta-aware grid wrapper.
    # Use a closure to capture num_pid_n via a dynamic launcher.
    # Trick: launch with a wrapper that computes NUM_PID_N from BLOCK_N.
    # Triton autotune passes meta; we need NUM_PID_N as a runtime arg. Use lambda-based.
    # We'll just pass N and BLOCK_N inside kernel; recompute num_pid_n in kernel using cdiv.
    gemm_rowmax_kernel[grid](
        x, weight, bias, partial,
        M, N, K, MAX_NUM_PID_N,  # placeholder, kernel will use its own
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        partial.stride(0), partial.stride(1),
    )
    # Get the actually-used BLOCK_N from the chosen config
    best_config = gemm_rowmax_kernel.best_config
    BLOCK_N_used = best_config.kwargs['BLOCK_N']
    num_pid_n_used = triton.cdiv(N, BLOCK_N_used)

    # Reduce over num_pid_n_used (only first num_pid_n_used rows are valid)
    BLOCK_M_R = 128
    BLOCK_PN = _next_pow2(num_pid_n_used)
    grid2 = (triton.cdiv(M, BLOCK_M_R),)
    reduce_max_gelu_kernel[grid2](
        partial, out,
        M, num_pid_n_used,
        partial.stride(0), partial.stride(1),
        BLOCK_M=BLOCK_M_R, BLOCK_PN=BLOCK_PN,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.gemm.weight.contiguous().cuda()
        bias = self.gemm.bias.contiguous().cuda()

        if self.max_dim == 1:
            # Fused GEMM + rowmax + (x - mean) + GELU
            out = gemm_rowmax_gelu(x, weight, bias)  # (B,)
            return out.unsqueeze(1)  # (B, 1)
        else:
            # max over dim=0 keepdim=True -> shape (1, N)
            y = torch.addmm(bias, x, weight.t())
            y = torch.max(y, dim=0, keepdim=True).values
            y = y - y.mean(dim=1, keepdim=True)
            out = torch.empty_like(y)
            n = y.numel()
            BLOCK = 256
            grid = (triton.cdiv(n, BLOCK),)
            gelu_kernel[grid](y, out, n, BLOCK_SIZE=BLOCK)
            return out