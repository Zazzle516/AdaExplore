import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gemm_groupnorm_min_kernel(
    x_ptr, w_ptr, b_ptr, gamma_ptr, beta_ptr, bias_ptr, out_ptr,
    M, K, N,
    num_groups, group_size,
    eps,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # One program per row (batch element)
    row = tl.program_id(0)
    if row >= M:
        return

    # Compute GEMM output for this row: y[n] = sum_k x[row,k] * w[n,k] + b[n]
    # w is stored as (N, K) from nn.Linear.weight
    # We'll process in tiles of BLOCK_N along N
    # Store results in a buffer of size N - we'll do it in chunks

    # Actually we need to compute all N outputs, do groupnorm, then min.
    # N can be large (8192). We allocate per-program intermediate via tl.zeros.
    # Process columns of N in tiles, accumulate over K.

    # For groupnorm, we need mean/var per group, so we need all N values.
    # We'll use a scratchpad? No - let's just compute in tiles of BLOCK_N,
    # storing into a register tile per group.

    # Simpler approach: compute gemm output into a temp buffer in global memory,
    # then do groupnorm + min in another kernel. But that's two kernels.

    # Alternative: since N=8192 fits, we can compute one tile of size N at a time
    # using a single program with BLOCK_N = N. Let BLOCK_N = N (constexpr).
    # That's 8192 floats = 32KB per program in registers/shared - feasible.

    offs_n = tl.arange(0, BLOCK_N)
    # Accumulate y[row, :] = x[row, :] @ w.T + b
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        # x_row: [BLOCK_K]
        x_row = tl.load(x_ptr + row * K + offs_k, mask=k_mask, other=0.0)
        # w: [N, K] -> we want w[offs_n, offs_k] -> shape [BLOCK_N, BLOCK_K]
        w_ptrs = w_ptr + offs_n[:, None] * K + offs_k[None, :]
        w_mask = (offs_n[:, None] < N) & (k_mask[None, :])
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        # acc += sum_k w_tile * x_row
        acc += tl.sum(w_tile * x_row[None, :], axis=1)

    # add bias
    b_vals = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b_vals

    # Group norm: reshape acc as [num_groups, group_size]
    # Compute mean and var per group
    # acc shape [BLOCK_N] = [num_groups * group_size]
    # Reshape via arithmetic: group_id = idx // group_size
    # We can reshape using tl.reshape
    acc2d = tl.reshape(acc, (num_groups, GROUP_SIZE))
    mean = tl.sum(acc2d, axis=1) / GROUP_SIZE  # [num_groups]
    diff = acc2d - mean[:, None]
    var = tl.sum(diff * diff, axis=1) / GROUP_SIZE
    rstd = 1.0 / tl.sqrt(var + eps)
    normed = diff * rstd[:, None]  # [num_groups, GROUP_SIZE]

    # affine
    gamma = tl.load(gamma_ptr + offs_n, mask=offs_n < N, other=0.0)
    beta = tl.load(beta_ptr + offs_n, mask=offs_n < N, other=0.0)
    normed_flat = tl.reshape(normed, (BLOCK_N,))
    y = normed_flat * gamma + beta

    # min along N (the feature dim)
    min_val = tl.min(y, axis=0)  # scalar

    # bias (shape [1, N, 1, 1]) -> after min and keepdim, output shape is [M, 1]
    # then added to bias broadcasting to [M, N, 1, 1]? Let's check:
    # x after min: [M, 1] (after squeezing the extra dims, but keepdim=True gives [M,1])
    # bias shape: (1, out_features, 1, 1)
    # x + bias broadcasts: [M, 1] + [1, N, 1, 1] -> [1, N, M, 1]? Actually:
    # [M, 1] is 2D, bias is 4D [1,N,1,1]. Broadcasting aligns from right:
    # [M, 1] -> treated as [1, 1, M, 1] for alignment
    # result: [1, N, M, 1]
    # We need to produce output of shape [1, N, M, 1] where each [0, n, m, 0] = min_val[m] + bias[n]

    # For this kernel, we just store min_val per row. Bias add happens outside.
    tl.store(out_ptr + row, min_val)


def fused_gemm_gn_min(x, weight, bias_lin, gamma, beta, num_groups, eps):
    M, K = x.shape
    N = weight.shape[0]
    group_size = N // num_groups

    out = torch.empty((M,), device=x.device, dtype=torch.float32)

    BLOCK_N = N  # must be power of 2 ideally; N=8192 is
    BLOCK_K = 64

    grid = (M,)
    gemm_groupnorm_min_kernel[grid](
        x, weight, bias_lin, gamma, beta, None, out,
        M, K, N,
        num_groups, group_size,
        eps,
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
        GROUP_SIZE=group_size,
        num_warps=8,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.gemm.weight.contiguous()
        bias_lin = self.gemm.bias.contiguous()
        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()
        eps = self.group_norm.eps

        M = x.shape[0]
        min_out = fused_gemm_gn_min(x, weight, bias_lin, gamma, beta, self.num_groups, eps)
        # min_out shape: [M]
        # Need to produce: min_val (shape [M,1]) + bias (shape [1,N,1,1])
        # Result: [1, N, M, 1] by broadcasting rules
        # [M,1] aligned as [1,1,M,1], bias [1,N,1,1] -> output [1,N,M,1]
        min_view = min_out.view(1, 1, M, 1)
        result = min_view + self.bias  # broadcasts to [1, N, M, 1]
        return result