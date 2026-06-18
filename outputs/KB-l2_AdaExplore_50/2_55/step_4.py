import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_gemm_pool_partial_kernel(
    x_ptr,        # (M, K)
    w_ptr,        # (N, K) row-major
    b_ptr,        # (N,)
    partial_ptr,  # (M, N_TILES)
    M, N, K,
    N_TILES,
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

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k
        k_mask = k_idx < K
        # x tile [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + offs_m[:, None] * K + k_idx[None, :]
        x_vals = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        # w tile [BLOCK_N, BLOCK_K]
        w_ptrs = w_ptr + offs_n[:, None] * K + k_idx[None, :]
        w_vals = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        # tl.dot: [BLOCK_M, BLOCK_K] x [BLOCK_K, BLOCK_N]
        acc += tl.dot(x_vals, tl.trans(w_vals))

    # bias add
    bias_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias_vals[None, :]

    # mask invalid n to -inf for pooling
    NEG_INF = float('-inf')
    acc = tl.where(n_mask[None, :], acc, NEG_INF)

    # max-pool kernel=2 along N: reshape [BLOCK_M, BLOCK_N/2, 2] then max axis=2
    acc_2d = tl.reshape(acc, (BLOCK_M, BLOCK_N // 2, 2))
    pooled = tl.max(acc_2d, axis=2)  # [BLOCK_M, BLOCK_N/2]
    pooled = tl.where(pooled == NEG_INF, 0.0, pooled)

    # partial sum across pooled axis -> [BLOCK_M]
    partial = tl.sum(pooled, axis=1)

    # store into partial[m, pid_n]
    out_ptrs = partial_ptr + offs_m * N_TILES + pid_n
    tl.store(out_ptrs, partial, mask=m_mask)


@triton.jit
def reduce_partial_kernel(
    partial_ptr,  # (M, N_TILES)
    out_ptr,      # (M,)
    M, N_TILES,
    scale,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < M:
        offs_t = tl.arange(0, BLOCK_T)
        acc = tl.zeros([BLOCK_T], dtype=tl.float32)
        for t_start in range(0, N_TILES, BLOCK_T):
            idx = t_start + offs_t
            mask = idx < N_TILES
            vals = tl.load(partial_ptr + pid * N_TILES + idx, mask=mask, other=0.0)
            acc += vals
        s = tl.sum(acc, axis=0) * scale
        tl.store(out_ptr + pid, s)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = float(scale_factor)
        self.matmul = nn.Linear(in_features, out_features)
        self.max_pool = nn.MaxPool1d(kernel_size)

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.matmul.weight.contiguous()
        B = self.matmul.bias.contiguous()
        M, K = x.shape
        N = W.shape[0]

        BLOCK_M = 16
        BLOCK_N = 128
        BLOCK_K = 64

        N_TILES = (N + BLOCK_N - 1) // BLOCK_N
        partial = torch.empty((M, N_TILES), device=x.device, dtype=torch.float32)

        grid = ((M + BLOCK_M - 1) // BLOCK_M, N_TILES)
        fused_gemm_pool_partial_kernel[grid](
            x, W, B, partial,
            M, N, K, N_TILES,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=3,
        )

        out = torch.empty((M,), device=x.device, dtype=torch.float32)
        # choose BLOCK_T as next pow2 >= N_TILES (capped) for simplicity
        BLOCK_T = 256
        reduce_partial_kernel[(M,)](
            partial, out, M, N_TILES,
            self.scale_factor,
            BLOCK_T=BLOCK_T,
            num_warps=4,
            num_stages=2,
        )
        return out