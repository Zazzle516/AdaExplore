import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_rowsum_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    row_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    m_mask = offs_m < M

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k_idx = k_start + offs_k
            k_mask = k_idx < K

            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk
            x_vals = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + k_idx[None, :] * stride_wk
            w_vals = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

            acc += tl.dot(x_vals, tl.trans(w_vals))

        # mask out-of-range n columns
        acc = tl.where(n_mask[None, :], acc, 0.0)
        row_acc += tl.sum(acc, axis=1)

    out_vals = row_acc * scale
    tl.store(out_ptr + offs_m * stride_om, out_vals, mask=m_mask)


def fused_matmul_div_sum_scale(x: torch.Tensor, weight: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    # x: (M, K), weight: (N, K)  -> out: (M, 1)
    # computes sum_n sum_k x[m,k] * weight[n,k] / 2 * scaling_factor
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    M, K = x.shape
    N, Kw = weight.shape
    assert K == Kw

    out = torch.empty((M, 1), device=x.device, dtype=torch.float32)
    scale = 0.5 * scaling_factor

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
    fused_gemm_rowsum_kernel[grid](
        x, weight, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0),
        scale,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = x.cuda().contiguous()
        return fused_matmul_div_sum_scale(x, self.weight, self.scaling_factor)