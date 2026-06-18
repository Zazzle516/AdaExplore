import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_rowsum_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < k_remaining)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x, tl.trans(w), allow_tf32=True)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # reduce along N within this tile
    row_partial = tl.sum(acc, axis=1)  # (BLOCK_M,)

    out_ptrs = out_ptr + pid_n * stride_on + offs_m * stride_om
    tl.store(out_ptrs, row_partial, mask=offs_m < M)


@triton.jit
def reduce_n_kernel(
    partial_ptr, bias_sum_ptr, out_ptr,
    M, GRID_N,
    stride_pn, stride_pm,
    BLOCK_GN: tl.constexpr,
):
    pid = tl.program_id(0)
    m = pid
    if m < M:
        offs = tl.arange(0, BLOCK_GN)
        mask = offs < GRID_N
        ptrs = partial_ptr + offs * stride_pn + m * stride_pm
        vals = tl.load(ptrs, mask=mask, other=0.0)
        s = tl.sum(vals, axis=0)
        bs = tl.load(bias_sum_ptr)
        tl.store(out_ptr + m, s + bs)


def fused_linear_rowsum(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2

    # Allocate partial with max possible grid_n based on smallest BLOCK_N in configs (64)
    max_grid_n = (N + 64 - 1) // 64
    partial = torch.zeros((max_grid_n, M), device=x.device, dtype=torch.float32)

    def grid(meta):
        gm = triton.cdiv(M, meta['BLOCK_M'])
        gn = triton.cdiv(N, meta['BLOCK_N'])
        return (gm * gn,)

    gemm_rowsum_kernel[grid](
        x, weight, partial,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        partial.stride(1), partial.stride(0),
    )

    # Determine actual grid_n used by chosen config
    best_cfg = gemm_rowsum_kernel.best_config
    actual_block_n = best_cfg.kwargs['BLOCK_N']
    actual_grid_n = (N + actual_block_n - 1) // actual_block_n

    bias_sum = bias.sum().to(torch.float32).reshape(())
    out = torch.empty((M,), device=x.device, dtype=torch.float32)

    BLOCK_GN = triton.next_power_of_2(actual_grid_n)
    reduce_n_kernel[(M,)](
        partial, bias_sum, out,
        M, actual_grid_n,
        partial.stride(0), partial.stride(1),
        BLOCK_GN=BLOCK_GN,
        num_warps=2,
    )
    return out.reshape(M, 1)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.linear.weight.contiguous()
        b = self.linear.bias.contiguous()
        out = fused_linear_rowsum(x, w, b)
        # The downstream ops (max over singleton, mean over singleton,
        # logsumexp over singleton dim twice) are all identity on a (M,1) tensor.
        return out