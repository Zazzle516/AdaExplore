import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_pool_partial_kernel(
    x_ptr, w_ptr, b_ptr, partial_ptr,
    M, N, K, NUM_GROUPS,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Computes y = x @ W^T + b, then pairwise max-pool along N (kernel=KERNEL_SIZE),
    then sums all pooled values along N -> partial[m] (per program tile contribution).

    x: [M, K]
    w: [N, K] (so W^T is [K, N]; we use w[n, k])
    b: [N]
    partial: [num_n_tiles, M]  (we atomic-add per (n_tile, m))

    Each program: one (m_tile, n_tile). BLOCK_N must be divisible by KERNEL_SIZE.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * K + offs_k[None, :]  # [BLOCK_N, BLOCK_K]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + offs_k) < K
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        # tl.dot: x [M,K] @ w^T [K,N] -> [M,N]
        acc += tl.dot(x, tl.trans(w), allow_tf32=True)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)  # [BLOCK_N]
    acc = acc + b_vals[None, :]

    # mask out-of-range N positions to -inf so they don't pollute max
    neg_inf = float('-inf')
    acc = tl.where(n_mask[None, :], acc, neg_inf)

    # reshape [BLOCK_M, BLOCK_N] -> [BLOCK_M, BLOCK_N/KERNEL_SIZE, KERNEL_SIZE]
    POOLED_N: tl.constexpr = BLOCK_N // KERNEL_SIZE
    acc3 = tl.reshape(acc, (BLOCK_M, POOLED_N, KERNEL_SIZE))
    pooled = tl.max(acc3, axis=2)  # [BLOCK_M, POOLED_N]

    # Replace -inf with 0 (these are positions beyond N anyway)
    pooled = tl.where(pooled == neg_inf, 0.0, pooled)

    row_sum = tl.sum(pooled, axis=1)  # [BLOCK_M]

    # atomic add to partial_ptr[m]
    tl.atomic_add(partial_ptr + offs_m, row_sum, mask=m_mask)


@triton.jit
def scale_kernel(
    partial_ptr, out_ptr, M,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    v = tl.load(partial_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, v * SCALE, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = float(scale_factor)
        self.matmul = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()
        device = x.device
        W = self.matmul.weight.to(device).contiguous()  # [N, K]
        b = self.matmul.bias.to(device).contiguous()    # [N]
        M = x.shape[0]
        K = self.in_features
        N = self.out_features

        assert N % self.kernel_size == 0

        partial = torch.zeros((M,), device=device, dtype=torch.float32)
        out = torch.empty((M,), device=device, dtype=torch.float32)

        num_groups = N // self.kernel_size

        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        gemm_pool_partial_kernel[grid](
            x, W, b, partial,
            M, N, K, num_groups,
            KERNEL_SIZE=self.kernel_size,
        )

        BLOCK = 128
        grid2 = (triton.cdiv(M, BLOCK),)
        scale_kernel[grid2](partial, out, M, SCALE=self.scale_factor, BLOCK=BLOCK)
        return out