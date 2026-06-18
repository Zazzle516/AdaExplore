import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_rowsum_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    row_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    m_mask = offs_m < M

    num_n_tiles = tl.cdiv(N, BLOCK_N)
    num_k_tiles = tl.cdiv(K, BLOCK_K)

    for n_idx in range(0, num_n_tiles):
        offs_n = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_idx in range(0, num_k_tiles):
            cur_k = k_idx * BLOCK_K + offs_k
            k_mask = cur_k < K

            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + cur_k[None, :] * stride_xk
            x_vals = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + cur_k[None, :] * stride_wk
            w_vals = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

            acc += tl.dot(x_vals, tl.trans(w_vals), allow_tf32=False)

        acc = tl.where(n_mask[None, :], acc, 0.0)
        row_acc += tl.sum(acc, axis=1)

    out = row_acc * scale
    tl.store(out_ptr + offs_m, out, mask=m_mask)


def fused_gemm_rowsum(x, w, scale):
    M, K = x.shape
    N, Kw = w.shape
    assert K == Kw
    x = x.contiguous()
    w = w.contiguous()
    out = torch.empty((M,), device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
    fused_gemm_rowsum_kernel[grid](
        x, w, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        scale,
    )
    return out.view(M, 1)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = float(scaling_factor)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        scale = 0.5 * self.scaling_factor
        return fused_gemm_rowsum(x, self.weight, scale)