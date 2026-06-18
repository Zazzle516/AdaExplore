import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_rowsum_kernel(
    X_ptr, Wt_ptr, Bsum_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = Wt_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Sum across N tile
    acc = tl.where(mask_n[None, :], acc, 0.0)
    row_partial = tl.sum(acc, axis=1)

    # Add bias-sum contribution for this N tile (only once per pid_m, so let pid_n=0 add it)
    # Actually we use atomic_add for partial, and add bias separately via atomic too.
    # Simpler: include the bias sum for this tile in row_partial.
    b_sum = tl.load(Bsum_ptr + pid_n)
    # b_sum is scalar for this N tile; add as broadcast over rows
    row_partial = row_partial + tl.where(mask_m, b_sum, 0.0)
    # Wait — bias should be added once total, not per row tile. Since each (pid_m, pid_n)
    # adds b_sum for its tile to its rows, summing over pid_n gives total bias sum per row.
    # But pid_m iterates rows, and each pid_n contributes b_sum_n to that row. Sum over pid_n
    # of b_sum_n = total bias sum. So this is correct.

    tl.atomic_add(Out_ptr + offs_m, row_partial, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features

        # Pre-transpose weight to (K, N) contiguous
        with torch.no_grad():
            Wt = self.linear.weight.detach().t().contiguous().cuda()
        self.register_buffer('Wt', Wt)

    def forward(self, x):
        x = x.contiguous().cuda()
        Wt = self.Wt
        B = self.linear.bias.contiguous().cuda()
        M, K = x.shape
        N = Wt.shape[1]

        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 32

        # Precompute per-tile bias sums
        n_tiles = (N + BLOCK_N - 1) // BLOCK_N
        # Pad B to multiple of BLOCK_N then sum per tile
        pad = n_tiles * BLOCK_N - N
        if pad > 0:
            B_padded = torch.cat([B, torch.zeros(pad, device=B.device, dtype=B.dtype)])
        else:
            B_padded = B
        Bsum = B_padded.view(n_tiles, BLOCK_N).sum(dim=1).contiguous()

        out = torch.zeros((M,), device=x.device, dtype=torch.float32)

        grid = (triton.cdiv(M, BLOCK_M), n_tiles)
        gemm_rowsum_kernel[grid](
            x, Wt, Bsum, out,
            M, N, K,
            x.stride(0), x.stride(1),
            Wt.stride(0), Wt.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=4,
        )

        return out.view(M, 1)