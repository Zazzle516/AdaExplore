import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_rowsum_kernel(
    x_ptr, wt_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = wt_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for k_idx in range(0, num_k_tiles):
        cur_k = k_idx * BLOCK_K + offs_k
        k_mask = cur_k < K

        x_vals = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w_vals = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc += tl.dot(x_vals, w_vals, allow_tf32=True)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    acc = tl.where(n_mask[None, :], acc, 0.0)
    row_partial = tl.sum(acc, axis=1) * scale

    # atomic add to per-row output
    tl.atomic_add(out_ptr + offs_m, row_partial, mask=m_mask)


def fused_gemm_rowsum(x, wt, scale):
    M, K = x.shape
    Kw, N = wt.shape
    assert K == Kw
    x = x.contiguous()
    out = torch.zeros((M,), device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    fused_gemm_rowsum_kernel[grid](
        x, wt, out,
        M, N, K,
        x.stride(0), x.stride(1),
        wt.stride(0), wt.stride(1),
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
        # Pre-transpose weight: (hidden_size, input_size) -> (input_size, hidden_size)
        # which corresponds to weight.T in (K, N) layout for the GEMM x @ weight.T
        self.register_buffer('_weight_t', None, persistent=False)

    def _get_weight_t(self):
        wt = self.weight.t().contiguous()
        return wt

    def forward(self, x):
        scale = 0.5 * self.scaling_factor
        wt = self.weight.t().contiguous()
        return fused_gemm_rowsum(x, wt, scale)