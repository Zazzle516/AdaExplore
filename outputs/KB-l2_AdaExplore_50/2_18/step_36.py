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
    num_tiles_m, num_tiles_n, total_tiles,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    
    for tile_id in range(pid, total_tiles, NUM_SMS):
        pid_m = tile_id // num_tiles_n
        pid_n = tile_id % num_tiles_n
        
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        
        mask_m = offs_m < M
        mask_n = offs_n < N
        
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for k in range(0, K, BLOCK_K):
            k_remaining = K - k
            mask_k = offs_k < k_remaining
            x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc += tl.dot(x, w, allow_tf32=True)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk
        
        # row-reduce over BLOCK_N
        row_partial = tl.sum(acc, axis=1)  # (BLOCK_M,)
        
        # atomic add into out[offs_m]
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

        # W is (N, K), we want x @ W.T => for tl.dot we need W as (K, N) layout
        # W.stride: stride_wn=K (row), stride_wk=1
        # accessing W[n, k] = w_ptr + n*K + k. For (K, N) view: w_ptr + k*1 + n*K, so stride_k=1, stride_n=K
        
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 32

        num_tiles_m = (M + BLOCK_M - 1) // BLOCK_M
        num_tiles_n = (N + BLOCK_N - 1) // BLOCK_N
        total_tiles = num_tiles_m * num_tiles_n

        # init output with bias.sum() so we can atomic add
        bias_sum = b.sum().to(torch.float32)
        out = torch.full((M,), bias_sum.item(), device=x.device, dtype=torch.float32)

        NUM_SMS = 128  # 4090 has 128 SMs
        grid = (NUM_SMS,)

        gemm_rowsum_persistent_kernel[grid](
            x, W, out,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            num_tiles_m, num_tiles_n, total_tiles,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            NUM_SMS=NUM_SMS,
            num_warps=4, num_stages=3,
        )

        return out.reshape(M, 1)