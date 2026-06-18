import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def col_sum_kernel(
    w_ptr, colsum_ptr,
    N, K,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Each program reduces a BLOCK_K slice of columns (over all N rows)
    pid = tl.program_id(0)
    k_start = pid * BLOCK_K
    offs_k = k_start + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        # w is (N, K) row-major
        ptrs = w_ptr + offs_n[:, None] * K + offs_k[None, :]
        mask = mask_n[:, None] & mask_k[None, :]
        vals = tl.load(ptrs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
    tl.store(colsum_ptr + offs_k, acc, mask=mask_k)


@triton.jit
def row_dot_kernel(
    x_ptr, c_ptr, out_ptr,
    M, K,
    SCALE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG
    offs = tl.arange(0, BLOCK_K)

    for i in tl.static_range(ROWS_PER_PROG):
        row = row_start + i
        row_valid = row < M
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            cur = k_start + offs
            mask = (cur < K) & row_valid
            xv = tl.load(x_ptr + row * K + cur, mask=mask, other=0.0)
            cv = tl.load(c_ptr + cur, mask=cur < K, other=0.0)
            acc += xv * cv
        s = tl.sum(acc, axis=0) * SCALE
        tl.store(out_ptr + row, s, mask=row_valid)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = float(scaling_factor)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        x = x.contiguous()
        w = self.weight.contiguous()
        N, K = w.shape  # hidden_size, input_size
        M = x.shape[0]

        scale = self.scaling_factor * 0.5

        # Step 1: compute column sums of W at runtime (N, K) -> (K)
        colsum = torch.empty((K,), device=x.device, dtype=w.dtype)
        BLOCK_K_CS = 256
        BLOCK_N_CS = 128
        grid_cs = (triton.cdiv(K, BLOCK_K_CS),)
        col_sum_kernel[grid_cs](
            w, colsum,
            N, K,
            BLOCK_K=BLOCK_K_CS,
            BLOCK_N=BLOCK_N_CS,
            num_warps=4,
            num_stages=3,
        )

        # Step 2: per-row dot product with colsum, scale, store
        out = torch.empty((M, 1), device=x.device, dtype=x.dtype)
        out_flat = out.view(M)

        ROWS_PER_PROG = 2
        BLOCK_K_RD = 2048
        grid_rd = (triton.cdiv(M, ROWS_PER_PROG),)
        row_dot_kernel[grid_rd](
            x, colsum, out_flat,
            M, K,
            SCALE=scale,
            BLOCK_K=BLOCK_K_RD,
            ROWS_PER_PROG=ROWS_PER_PROG,
            num_warps=8,
            num_stages=3,
        )
        return out