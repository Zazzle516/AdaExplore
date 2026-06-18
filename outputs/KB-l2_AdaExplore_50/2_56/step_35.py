import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_matmul_sigmoid_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
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

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K
        x = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # add bias
    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]
    # sigmoid
    acc = tl.sigmoid(acc)
    # mask out invalid n
    acc = tl.where(mask_n[None, :], acc, 0.0)
    # reduce along N within this tile
    row_partial = tl.sum(acc, axis=1)
    # atomic add into out[offs_m]
    tl.atomic_add(out_ptr + offs_m, row_partial, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.linear.weight  # [hidden_size, input_size]
        b = self.linear.bias    # [hidden_size]

        M, K = x.shape
        N = W.shape[0]

        # W transposed view: [K, N] via stride manipulation
        Wt = W.t()  # [input_size, hidden_size]
        # ensure contiguous-ish access
        W_for_kernel = Wt.contiguous() if not Wt.is_contiguous() else Wt

        out = torch.zeros(M, device=x.device, dtype=torch.float32)

        BLOCK_M = 32
        BLOCK_N = 64
        BLOCK_K = 64

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        fused_matmul_sigmoid_sum_kernel[grid](
            x, W_for_kernel, b, out,
            M, N, K,
            x.stride(0), x.stride(1),
            W_for_kernel.stride(0), W_for_kernel.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        return out.unsqueeze(1)