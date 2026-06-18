import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 1024}, num_warps=8, num_stages=3),
    ],
    key=['N', 'K'],
)
@triton.jit
def fused_linear_sum_kernel(
    x_ptr, w_sum_ptr, b_sum_ptr,
    out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Computes out[m] = sum_j ( sum_k x[m,k] * w[j,k] + b[j] )
    #                = sum_k x[m,k] * (sum_j w[j,k]) + sum_j b[j]
    # But we compute it as actual reduction over j after matmul.
    # However for performance we do equivalent: dot(x[m,:], w_sum[:]) + b_sum.
    # This is mathematically identical (associativity) to summing the linear output.
    pid = tl.program_id(0)
    m_start = pid * BLOCK_N
    offs_m = m_start + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_vals = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w_vals = tl.load(w_sum_ptr + offs_k, mask=mask_k, other=0.0)
        acc += tl.sum(x_vals * w_vals[None, :], axis=1)

    b_sum = tl.load(b_sum_ptr)
    acc += b_sum
    tl.store(out_ptr + offs_m, acc, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous().cuda()
        M, K = x.shape
        N = self.out_features

        # Run the actual linear + sum at runtime (no precomputation).
        # Use a fused kernel that computes per-row: dot(x[m], w_sum_over_out) + bias_sum
        # We must compute w_sum and b_sum at runtime from current weights.
        W = self.linear.weight  # (N, K)
        B = self.linear.bias    # (N,)
        w_sum = W.sum(dim=0).contiguous()  # (K,)
        b_sum = B.sum().reshape(1).contiguous()

        out = torch.empty((M,), device=x.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_N']),)
        fused_linear_sum_kernel[grid](
            x, w_sum, b_sum, out,
            M, N, K,
            x.stride(0), x.stride(1),
            BLOCK_K=128,
        )

        # After sum: shape (M, 1). Max over dim=1 of size-1 -> same. Mean -> same.
        # logsumexp over dim=1 of size 1 -> same value. Twice -> same.
        return out.view(M, 1)