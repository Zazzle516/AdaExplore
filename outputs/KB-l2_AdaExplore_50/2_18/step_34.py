import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_rowsum_kernel(
    x_ptr, w_ptr, bias_sum_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M

    row_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    num_n_tiles = (N + BLOCK_N - 1) // BLOCK_N
    for pid_n in range(0, num_n_tiles):
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

        for k in range(0, K, BLOCK_K):
            k_mask = (k + offs_k) < K
            x = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=mask_n[:, None] & k_mask[None, :], other=0.0)
            acc += tl.dot(x, tl.trans(w), allow_tf32=True)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk

        # Zero out masked-N lanes before sum (they may contain garbage from tl.dot when N is not a multiple)
        acc = tl.where(mask_n[None, :], acc, 0.0)
        row_acc += tl.sum(acc, axis=1)

    bias_s = tl.load(bias_sum_ptr)
    row_acc = row_acc + bias_s
    tl.store(out_ptr + offs_m, row_acc, mask=mask_m)


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

        bias_sum = b.to(torch.float32).sum().reshape(1)
        out = torch.empty((M,), device=x.device, dtype=torch.float32)

        grid = lambda META: ((M + META['BLOCK_M'] - 1) // META['BLOCK_M'],)

        gemm_rowsum_kernel[grid](
            x, W, bias_sum, out,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
        )

        return out.reshape(M, 1)