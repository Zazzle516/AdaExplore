import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K', 'SPLIT_K'],
)
@triton.jit
def fused_linear_sum_splitk_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # split K across pid_k
    k_per_split = (K + SPLIT_K - 1) // SPLIT_K
    k_begin = pid_k * k_per_split
    k_end = tl.minimum(k_begin + k_per_split, K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(k_begin, k_end, BLOCK_K):
        k_idx = k_start + offs_k
        mask_k = k_idx < k_end
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + k_idx[None, :] * stride_wk
        w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w), allow_tf32=True)

    # add bias only in the first split (pid_k==0) so atomic sum totals correctly
    if pid_k == 0:
        b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        acc = acc + b[None, :]

    acc = tl.where(mask_n[None, :], acc, 0.0)
    row_partial = tl.sum(acc, axis=1)

    # atomic add into out[offs_m]
    tl.atomic_add(out_ptr + offs_m, row_partial, mask=mask_m)


def fused_linear_rowsum(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    out = torch.zeros(M, device=x.device, dtype=torch.float32)

    # Choose SPLIT_K based on K to keep enough parallelism
    SPLIT_K = 4 if K >= 4096 else (2 if K >= 1024 else 1)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
        SPLIT_K,
    )
    fused_linear_sum_splitk_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        SPLIT_K=SPLIT_K,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.linear.weight.contiguous()
        b = self.linear.bias.contiguous()
        s = fused_linear_rowsum(x, w, b)  # (M,)
        return s.unsqueeze(1)