import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_rowsum_persistent_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    NUM_TILES_M: tl.constexpr,
    NUM_TILES_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_PROGS: tl.constexpr,
):
    pid = tl.program_id(0)
    total_tiles = NUM_TILES_M * NUM_TILES_N

    for tile_id in range(pid, total_tiles, NUM_PROGS):
        pid_m = tile_id // NUM_TILES_N
        pid_n = tile_id % NUM_TILES_N

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

        for k in range(0, K, BLOCK_K):
            x = tl.load(x_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < K - k), other=0.0)
            w = tl.load(w_ptrs, mask=mask_n[:, None] & (offs_k[None, :] < K - k), other=0.0)
            acc += tl.dot(x, tl.trans(w), allow_tf32=True)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk

        row_partial = tl.sum(acc, axis=1)
        tl.atomic_add(out_ptr + offs_m, row_partial, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.linear.weight.contiguous()  # (N, K)
        b = self.linear.bias.contiguous()    # (N,)

        M, K = x.shape
        N = W.shape[0]

        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 32

        NUM_TILES_M = (M + BLOCK_M - 1) // BLOCK_M
        NUM_TILES_N = (N + BLOCK_N - 1) // BLOCK_N

        bias_sum = b.sum().to(torch.float32)
        out = bias_sum.expand(M).contiguous().clone()

        NUM_PROGS = 128

        gemm_rowsum_persistent_kernel[(NUM_PROGS,)](
            x, W, out,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            NUM_TILES_M=NUM_TILES_M,
            NUM_TILES_N=NUM_TILES_N,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            NUM_PROGS=NUM_PROGS,
            num_warps=4, num_stages=3,
        )

        return out.reshape(M, 1)