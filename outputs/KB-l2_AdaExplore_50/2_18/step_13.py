import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_rowsum_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M

    # K range for this program
    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start_split = pid_k * k_per_split
    k_end_split = tl.minimum(k_start_split + k_per_split, K)

    row_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # loop over N tiles
    for n_start in range(0, N, BLOCK_N):
        cur_offs_n = n_start + offs_n
        mask_n = cur_offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        x_base = X_ptr + offs_m[:, None] * stride_xm
        w_base = W_ptr + cur_offs_n[:, None] * stride_wn

        for k_off in range(k_start_split, k_end_split, BLOCK_K):
            cur_k = k_off + offs_k
            mask_k = cur_k < k_end_split
            x = tl.load(x_base + cur_k[None, :] * stride_xk,
                        mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            w = tl.load(w_base + cur_k[None, :] * stride_wk,
                        mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            acc += tl.dot(x, tl.trans(w))

        # only the first split adds bias (avoid double counting)
        if pid_k == 0:
            b = tl.load(B_ptr + cur_offs_n, mask=mask_n, other=0.0)
            acc = acc + b[None, :]

        acc = tl.where(mask_n[None, :], acc, 0.0)
        row_acc += tl.sum(acc, axis=1)

    tl.atomic_add(Out_ptr + offs_m, row_acc, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.linear.weight.contiguous()
        B = self.linear.bias.contiguous()
        M, K = x.shape
        N = W.shape[0]

        out = torch.zeros((M,), device=x.device, dtype=torch.float32)

        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 64
        SPLIT_K = 4

        grid = (triton.cdiv(M, BLOCK_M), SPLIT_K)
        gemm_rowsum_kernel[grid](
            x, W, B, out,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            SPLIT_K=SPLIT_K,
            num_warps=4, num_stages=3,
        )

        return out.view(M, 1)