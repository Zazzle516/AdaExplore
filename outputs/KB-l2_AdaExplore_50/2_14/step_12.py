import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_rowsum_kernel(
    A_ptr, B_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Each tile contributes sum(acc, axis=1) * scale to out[m]
    row_partial = tl.sum(acc, axis=1) * scale
    tl.atomic_add(out_ptr + offs_m, row_partial, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.scaling_factor = scaling_factor
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))

    def forward(self, x):
        x = x.contiguous()
        M, K = x.shape
        N = self.hidden_size
        # weight is (hidden_size, input_size) = (N, K); we need W.T => (K, N)
        # Use weight directly with stride trick: B is W.T, so B[k, n] = W[n, k]
        W = self.weight
        # B_ptr with stride_bk = W.stride(1), stride_bn = W.stride(0)
        out = torch.zeros(M, device=x.device, dtype=torch.float32)

        # combined scale: /2 then * scaling_factor
        scale = 0.5 * self.scaling_factor

        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N, META['BLOCK_N']),
        )

        gemm_rowsum_kernel[grid](
            x, W, out,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(1), W.stride(0),  # B = W.T
            scale,
        )

        return out.view(M, 1)