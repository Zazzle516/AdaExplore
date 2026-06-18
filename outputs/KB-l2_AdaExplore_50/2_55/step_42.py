import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_pool_sum_kernel(
    x_ptr,        # [M, K]
    wt_ptr,       # [K, N]  (weight transposed, contiguous)
    b_ptr,        # [N]
    out_ptr,      # [M]
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    scale_factor: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = wt_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remain = K - k
        mask_k = offs_k < k_remain
        a = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # add bias
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # mask out-of-N columns to -inf so they don't poison max
    acc = tl.where(mask_n[None, :], acc, float('-inf'))

    # pool along N axis: reshape BLOCK_N -> (BLOCK_N/KERNEL_SIZE, KERNEL_SIZE) and max
    POOLED: tl.constexpr = BLOCK_N // KERNEL_SIZE
    acc_3d = tl.reshape(acc, (BLOCK_M, POOLED, KERNEL_SIZE))
    pooled = tl.max(acc_3d, axis=2)  # [BLOCK_M, POOLED]

    # sum across pooled
    partial = tl.sum(pooled, axis=1) * scale_factor  # [BLOCK_M]

    # atomic add into out[offs_m]
    tl.atomic_add(out_ptr + offs_m, partial, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = int(kernel_size)
        self.scale_factor = float(scale_factor)
        self.matmul = nn.Linear(in_features, out_features)
        # Pre-transpose weight for contiguous K-axis loads. Store as [K, N] contiguous.
        with torch.no_grad():
            wt = self.matmul.weight.detach().t().contiguous()
        self.register_buffer('weight_t', wt)

    def forward(self, x):
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()
        M, K = x.shape
        N = self.out_features

        wt = self.weight_t
        if wt.device != x.device:
            wt = wt.to(x.device)
            self.weight_t = wt
        bias = self.matmul.bias
        if bias.device != x.device:
            bias = bias.to(x.device)

        out = torch.zeros(M, device=x.device, dtype=torch.float32)

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )
        fused_gemm_pool_sum_kernel[grid](
            x, wt, bias, out,
            M, N, K,
            x.stride(0), x.stride(1),
            wt.stride(0), wt.stride(1),
            scale_factor=self.scale_factor,
            KERNEL_SIZE=self.kernel_size,
        )
        return out