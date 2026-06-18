import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
    ],
    key=['N', 'K'],
)
@triton.jit
def _fused_linear_sigmoid_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    row_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + offs_k
            mask_k = k_offs < K

            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
            x_vals = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + k_offs[None, :] * stride_wk
            w_vals = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

            acc += tl.dot(x_vals, tl.trans(w_vals))

        b_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        acc = acc + b_vals[None, :]

        sig = 1.0 / (1.0 + tl.exp(-acc))
        sig = tl.where(mask_n[None, :], sig, 0.0)

        row_acc += tl.sum(sig, axis=1)

    tl.store(out_ptr + offs_m, row_acc, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        x = x.contiguous().cuda()
        M, K = x.shape
        N = self.hidden_size
        W = self.linear.weight.contiguous()
        B = self.linear.bias.contiguous()

        out = torch.empty((M,), device=x.device, dtype=torch.float32)

        BLOCK_M = 128 if M >= 128 else 64
        # ensure BLOCK_M is at least 16 for tl.dot
        if M <= 16:
            BLOCK_M = 16
        elif M <= 32:
            BLOCK_M = 32
        elif M <= 64:
            BLOCK_M = 64
        else:
            BLOCK_M = 128

        grid = lambda meta: (triton.cdiv(M, BLOCK_M),)

        _fused_linear_sigmoid_sum_kernel[grid](
            x, W, B, out,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            BLOCK_M=BLOCK_M,
        )

        return out.view(M, 1)