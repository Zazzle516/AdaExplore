import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_sigmoid_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    for k in range(0, K, BLOCK_K):
        k_mask = offs_k[None, :] < (K - k)
        x = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask, other=0.0)
        w = tl.load(w_ptrs, mask=mask_n[:, None] & k_mask, other=0.0)
        acc += tl.dot(x, tl.trans(w), allow_tf32=True)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += b[None, :]

    sig = tl.sigmoid(acc)
    sig = tl.where(mask_m[:, None] & mask_n[None, :], sig, 0.0)
    row_sum = tl.sum(sig, axis=1)

    tl.atomic_add(out_ptr + offs_m, row_sum, mask=mask_m)


def fused_linear_sigmoid_sum(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2

    out = torch.zeros(M, device=x.device, dtype=torch.float32)

    BLOCK_M = 32 if M >= 32 else triton.next_power_of_2(M)
    if BLOCK_M < 16:
        BLOCK_M = 16

    grid = lambda meta: (triton.cdiv(M, BLOCK_M), triton.cdiv(N, meta['BLOCK_N']))

    fused_linear_sigmoid_sum_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        BLOCK_M=BLOCK_M,
    )
    return out.view(M, 1)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.linear.weight.contiguous().cuda()
        b = self.linear.bias.contiguous().cuda()
        return fused_linear_sigmoid_sum(x, w, b)