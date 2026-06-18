import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_K': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_K': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 512}, num_warps=8, num_stages=2),
    ],
    key=['K', 'N'],
)
@triton.jit
def linear_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_K: tl.constexpr,
):
    # One program per row m. Computes sum_n( sum_k x[m,k]*w[n,k] + b[n] )
    # = sum_k x[m,k] * (sum_n w[n,k]) + sum_n b[n]
    # But to avoid the safety contract issue, we compute the full linear then sum.
    # We accumulate per-row: for each k chunk, load x[m, k_chunk], and accumulate
    # the dot with sum over n of w[n, k_chunk].
    # However to keep linear executing at runtime, we instead iterate over n.
    pid = tl.program_id(0)
    m = pid

    # Accumulate sum over n of (dot(x[m,:], w[n,:]) + b[n])
    # Equivalently we compute y[m] = sum_n linear(x,w,b)[m,n]
    # Approach: tile K. For each K tile, load x[m, k_tile] (vector size BLOCK_K),
    # then for that k tile, we need sum_n w[n, k_tile]. That would be a weight
    # reduction at runtime (still a runtime op), let's just do straightforward:
    # iterate K in tiles, compute partial = x[m,k_tile]; then we need to do
    # sum_n (x . w[n,:]) which requires touching all w. We'll loop over n in
    # blocks too.
    
    # Simpler: do a full per-row matvec accumulation: for each k, contribution
    # to output_n is x[m,k]*w[n,k]. Sum over n: x[m,k] * sum_n w[n,k].
    # This requires reducing w over n at runtime - we just do it on the fly.
    
    acc = 0.0
    # Loop over K
    k_start = 0
    for k_off in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_off * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        # Load x[m, offs_k]
        x_vals = tl.load(x_ptr + m * stride_xm + offs_k * stride_xk, mask=k_mask, other=0.0)
        # Sum over n of w[n, offs_k] -> shape [BLOCK_K]
        # We accumulate by looping over n
        w_col_sum = tl.zeros([BLOCK_K], dtype=tl.float32)
        for n_idx in range(0, N):
            w_row = tl.load(w_ptr + n_idx * stride_wn + offs_k * stride_wk, mask=k_mask, other=0.0)
            w_col_sum += w_row
        acc += tl.sum(x_vals * w_col_sum)
    
    # Add sum of bias
    b_sum = 0.0
    for n_idx in range(0, N):
        b_sum += tl.load(b_ptr + n_idx)
    acc += b_sum
    
    tl.store(out_ptr + m, acc)


# The above is too slow due to the inner n loop. Let me write a proper two-stage version.


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_rowsum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Computes y[m, n_tile] = x[m,:] @ w[n_tile,:].T + b[n_tile]
    # Then accumulates sum over n_tile -> partial[m, pid_n]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < k_remaining)
        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x_block, tl.trans(w_block))
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
    
    # add bias
    b_vals = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += b_vals[None, :]
    
    # mask out invalid n elements before summing
    n_valid_mask = offs_n[None, :] < N
    acc = tl.where(n_valid_mask, acc, 0.0)
    
    # sum across N
    row_sum = tl.sum(acc, axis=1)  # [BLOCK_M]
    
    # atomic add into out[offs_m]
    m_mask = offs_m < M
    tl.atomic_add(out_ptr + offs_m, row_sum, mask=m_mask)


def fused_linear_sum(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    out = torch.zeros(M, device=x.device, dtype=torch.float32)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    matmul_bias_rowsum_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        x = x.contiguous()
        w = self.linear.weight.contiguous()
        b = self.linear.bias.contiguous()
        # Fused linear + sum over output features -> [M]
        y = fused_linear_sum(x, w, b)  # [M]
        # max over dim=1 with keepdim => still [M, 1] (single element)
        # mean over dim=1 keepdim => [M, 1]
        # logsumexp over dim=1 keepdim of single element => same value
        # twice => same
        return y.unsqueeze(1)