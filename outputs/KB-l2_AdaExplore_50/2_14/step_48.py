import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _gemm_rowsum_kernel(
    x_ptr, w_ptr, scratch_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_sm, stride_st,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_curr = k + offs_k
        mask_k = k_curr < K
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w))
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Mask out-of-range N columns to 0 before reducing
    acc = tl.where(mask_n[None, :], acc, 0.0)
    row_partial = tl.sum(acc, axis=1)  # (BLOCK_M,)

    # Write to scratch[pid_n, offs_m]
    out_ptrs = scratch_ptr + pid_n * stride_st + offs_m * stride_sm
    tl.store(out_ptrs, row_partial, mask=mask_m)


@triton.jit
def _reduce_scratch_kernel(
    scratch_ptr, out_ptr,
    M, NUM_TILES,
    stride_sm, stride_st,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_t = tl.arange(0, BLOCK_T)
    mask_m = offs_m < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for t_start in range(0, NUM_TILES, BLOCK_T):
        t = t_start + offs_t
        mask_t = t < NUM_TILES
        ptrs = scratch_ptr + t[None, :] * stride_st + offs_m[:, None] * stride_sm
        vals = tl.load(ptrs, mask=mask_m[:, None] & mask_t[None, :], other=0.0)
        acc += tl.sum(vals, axis=1)

    acc = acc * scale
    tl.store(out_ptr + offs_m, acc, mask=mask_m)


def _launch(x, weight, scaling_factor):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2

    x = x.contiguous()
    weight = weight.contiguous()

    # Determine grid using autotune meta
    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    # We need scratch with shape (num_n_tiles, M). num_n_tiles depends on autotuned BLOCK_N.
    # Allocate worst-case: use BLOCK_N=128 minimum. Actually need exact, so allocate per call after autotune.
    # Simpler: allocate based on smallest BLOCK_N from configs (128) — that's max num_n_tiles.
    max_num_n_tiles = triton.cdiv(N, 128)
    scratch = torch.empty((max_num_n_tiles, M), device=x.device, dtype=torch.float32)

    _gemm_rowsum_kernel[grid](
        x, weight, scratch,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        scratch.stride(1), scratch.stride(0),
    )

    # Determine actual num_n_tiles used
    # Get the chosen config
    best_config = _gemm_rowsum_kernel.best_config
    block_n = best_config.kwargs["BLOCK_N"]
    actual_num_n_tiles = triton.cdiv(N, block_n)

    out = torch.empty((M,), device=x.device, dtype=torch.float32)
    BLOCK_M2 = 128
    BLOCK_T = 32
    grid2 = (triton.cdiv(M, BLOCK_M2),)
    scale = scaling_factor * 0.5
    _reduce_scratch_kernel[grid2](
        scratch, out,
        M, actual_num_n_tiles,
        scratch.stride(1), scratch.stride(0),
        scale,
        BLOCK_M=BLOCK_M2, BLOCK_T=BLOCK_T,
    )

    return out.view(M, 1)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = x.cuda()
        return _launch(x, self.weight, self.scaling_factor)