import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_linear_rowsum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
    # Each program computes a BLOCK_M x BLOCK_N tile of x @ W^T + b,
    # then reduces along N within the tile and atomic_adds into out[m].
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m_mask = offs_m < M
    n_mask = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_idx = k + offs_k
        k_mask = k_idx < K
        x_vals = tl.load(x_ptrs + k * stride_xk,
                         mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w_vals = tl.load(w_ptrs + k * stride_wk,
                         mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc += tl.dot(x_vals, w_vals, allow_tf32=False)

    b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc += b_vals[None, :]

    acc = tl.where(n_mask[None, :], acc, 0.0)
    row_partial = tl.sum(acc, axis=1)

    tl.atomic_add(out_ptr + offs_m, row_partial, mask=m_mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous().cuda()
        M, K = x.shape
        N = self.out_features

        W = self.linear.weight.contiguous()  # (N, K)
        B = self.linear.bias.contiguous()    # (N,)

        out = torch.zeros((M,), device=x.device, dtype=torch.float32)

        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        _fused_linear_rowsum_kernel[grid](
            x, W, B, out,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # After row-sum: out is (M,). Subsequent ops:
        # max over dim=1 keepdim -> (M,1) (same value since size-1)
        # mean over dim=1 -> same
        # logsumexp over dim=1 (size 1) -> same value
        # logsumexp again -> same.
        # So final output is just out reshaped.
        return out.view(M, 1)