import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_gemm_pool_sum_kernel(
    x_ptr,      # (M, K) fp32
    w_ptr,      # (K, N) fp32 (pre-transposed)
    b_ptr,      # (N,) fp32
    out_ptr,    # (M,) fp32
    M, N, K,
    scale,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_start = pid_n * SPLIT_N
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M

    row_total = tl.zeros([BLOCK_M], dtype=tl.float32)

    # iterate over N tiles within this split
    for n_off in range(0, SPLIT_N, BLOCK_N):
        offs_n = n_start + n_off + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        x_base = x_ptr + offs_m[:, None] * stride_xm
        w_base = w_ptr + offs_n[None, :] * stride_wn

        for k_start in range(0, K, BLOCK_K):
            k_idx = k_start + offs_k
            k_mask = k_idx < K
            x_vals = tl.load(
                x_base + k_idx[None, :] * stride_xk,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            w_vals = tl.load(
                w_base + k_idx[:, None] * stride_wk,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(x_vals, w_vals, allow_tf32=True)

        # add bias
        bias_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
        acc = acc + bias_vals[None, :]
        # invalid n positions -> -inf so pool doesn't pick them
        acc = tl.where(n_mask[None, :], acc, float('-inf'))

        # maxpool kernel=2 along N
        acc_2d = tl.reshape(acc, (BLOCK_M, BLOCK_N // 2, 2))
        pooled = tl.max(acc_2d, axis=2)  # [BLOCK_M, BLOCK_N//2]
        pooled = tl.where(pooled == float('-inf'), 0.0, pooled)
        row_total += tl.sum(pooled, axis=1)

    row_total = row_total * scale
    # atomic add into out
    tl.atomic_add(out_ptr + offs_m, row_total, mask=m_mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = float(scale_factor)
        self.matmul = nn.Linear(in_features, out_features)
        self.max_pool = nn.MaxPool1d(kernel_size)
        # pre-transpose weight to (K, N) contiguous
        with torch.no_grad():
            wt = self.matmul.weight.detach().t().contiguous()
        self.register_buffer('weight_t', wt)

    def forward(self, x):
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()
        W = self.weight_t
        if W.device != x.device:
            W = W.to(x.device)
            self.weight_t = W
        B = self.matmul.bias.contiguous()
        if B.device != x.device:
            B = B.to(x.device)

        M, K = x.shape
        N = W.shape[1]

        out = torch.zeros((M,), device=x.device, dtype=torch.float32)

        BLOCK_M = 32
        BLOCK_N = 128
        BLOCK_K = 32
        SPLIT_N = 4096  # each program handles 4096 columns of N

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, SPLIT_N))

        fused_gemm_pool_sum_kernel[grid](
            x, W, B, out,
            M, N, K,
            self.scale_factor,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            SPLIT_N=SPLIT_N,
            num_warps=4,
            num_stages=3,
        )
        return out