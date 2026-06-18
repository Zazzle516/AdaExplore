import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 1024}, num_warps=8, num_stages=2),
    ],
    key=['K', 'N'],
)
@triton.jit
def fused_linear_sigmoid_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    K, N,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # One program per row m
    m = tl.program_id(0)

    x_row_ptr = x_ptr + m * stride_xm

    acc_sum = 0.0

    # iterate over output features in blocks of BLOCK_N
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        # Accumulator for the BLOCK_N outputs
        out_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            # Load x[m, k_start:k_start+BLOCK_K]: shape (BLOCK_K,)
            x_vals = tl.load(x_row_ptr + offs_k * stride_xk, mask=k_mask, other=0.0)

            # Load W[offs_n, offs_k] : shape (BLOCK_N, BLOCK_K)
            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
            w_mask = n_mask[:, None] & k_mask[None, :]
            w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

            # Matvec: (BLOCK_N, BLOCK_K) @ (BLOCK_K,) -> (BLOCK_N,)
            out_acc += tl.sum(w_vals * x_vals[None, :], axis=1)

        # Add bias
        b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
        out_acc = out_acc + b_vals

        # Sigmoid
        sig = tl.sigmoid(out_acc)
        sig = tl.where(n_mask, sig, 0.0)

        acc_sum += tl.sum(sig, axis=0)

    tl.store(out_ptr + m, acc_sum)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.linear.weight.contiguous().cuda()  # (N, K)
        b = self.linear.bias.contiguous().cuda()    # (N,)

        M, K = x.shape
        N = W.shape[0]

        out = torch.empty((M, 1), device=x.device, dtype=torch.float32)

        BLOCK_N = 128

        grid = (M,)
        fused_linear_sigmoid_sum_kernel[grid](
            x, W, b, out,
            K, N,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            BLOCK_N=BLOCK_N,
        )
        return out