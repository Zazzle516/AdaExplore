import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_rowsum_kernel(
    x_ptr, w_ptr, partial_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # row sum * 0.5
    row_sum = tl.sum(acc, axis=1) * 0.5  # (BLOCK_M,)
    # atomic add into partial[m]
    tl.atomic_add(partial_ptr + offs_m, row_sum, mask=mask_m)


@triton.jit
def scale_kernel(in_ptr, out_ptr, M, scaling_factor: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x * scaling_factor, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = float(scaling_factor)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        x = x.contiguous()
        M, K = x.shape
        N = self.hidden_size

        # weight is (hidden_size, input_size) = (N, K). We want W of shape (K, N) for GEMM.
        # Use weight.T (view) - has stride (1, K) which is column-major for B; that's fine.
        W = self.weight  # (N, K)
        # Use as B with B[k, n] = W[n, k], i.e. ptrs with stride_wk=1, stride_wn=K
        partial = torch.zeros(M, device=x.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
        gemm_rowsum_kernel[grid](
            x, W, partial,
            M, N, K,
            x.stride(0), x.stride(1),
            1, W.stride(0),  # B is W.T: stride_wk=1, stride_wn=K (=W.stride(0))
        )

        out = torch.empty(M, 1, device=x.device, dtype=x.dtype)
        BLOCK = 256
        grid2 = (triton.cdiv(M, BLOCK),)
        scale_kernel[grid2](partial, out, M, self.scaling_factor, BLOCK=BLOCK)
        return out