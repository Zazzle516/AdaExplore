import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _row_sum_gemm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M

    # Per-row accumulator for sum over N
    row_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    num_n_tiles = tl.cdiv(N, BLOCK_N)

    for nt in range(0, num_n_tiles):
        offs_n = nt * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + offs_k
            k_mask = k_offs < K

            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
            x_tile = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + k_offs[None, :] * stride_wk
            w_tile = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

            acc += tl.dot(x_tile, tl.trans(w_tile), allow_tf32=True)

        b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
        acc += b_vals[None, :]
        acc = tl.where(n_mask[None, :], acc, 0.0)

        row_acc += tl.sum(acc, axis=1)

    tl.atomic_add(out_ptr + offs_m, row_acc, mask=m_mask)


def fused_linear_rowsum(x, W, b):
    M, K = x.shape
    N = W.shape[0]
    out = torch.zeros(M, device=x.device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),)
    _row_sum_gemm_kernel[grid](
        x, W, b, out,
        M, N, K,
        x.stride(0), x.stride(1),
        W.stride(0), W.stride(1),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.linear.weight.contiguous()
        b = self.linear.bias.contiguous()
        row_sum = fused_linear_rowsum(x, W, b)  # (M,)
        # max/mean/logsumexp/logsumexp along dim=1 with size 1 are all identity
        out = row_sum.unsqueeze(1)
        return out