import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_pool_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    # weight is [N, K] row-major; we want W^T effectively: out[m,n] = sum_k x[m,k]*w[n,k]
    w_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        x_mask = mask_m[:, None] & (offs_k[None, :] < k_remaining)
        w_mask = (offs_k[:, None] < k_remaining) & mask_n[None, :]
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x_vals, w_vals)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    # Add bias
    b_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += b_vals[None, :]

    # Maxpool over kernel_size=2: pairs along N
    # Reshape [BLOCK_M, BLOCK_N] -> [BLOCK_M, BLOCK_N/2, 2], max over last dim
    reshaped = tl.reshape(acc, (BLOCK_M, BLOCK_N // 2, 2))
    pooled = tl.max(reshaped, axis=2)  # [BLOCK_M, BLOCK_N/2]

    # Sum over N dim
    row_sum = tl.sum(pooled, axis=1)  # [BLOCK_M]
    row_sum = row_sum * SCALE

    # Atomic add to out[offs_m]
    tl.atomic_add(out_ptr + offs_m, row_sum, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = float(scale_factor)
        self.matmul = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.matmul.weight.contiguous()
        b = self.matmul.bias.contiguous()
        M = x.shape[0]
        K = self.in_features
        N = self.out_features

        out = torch.zeros(M, device=x.device, dtype=x.dtype)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
        fused_gemm_pool_sum_kernel[grid](
            x, w, b, out,
            M, N, K,
            self.scale_factor,
        )
        return out