import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 1}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 1}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 1}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 1}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 1}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 1}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 1}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 1}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 1}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
    reset_to_zero=['out_ptr'],
)
@triton.jit
def _fused_gemm_sigmoid_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        mask_k = k_offs < K

        x_vals = tl.load(x_ptrs + k_start * stride_xk,
                         mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w_vals = tl.load(w_ptrs + k_start * stride_wk,
                         mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc += tl.dot(x_vals, w_vals)

    b_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b_vals[None, :]

    sig = 1.0 / (1.0 + tl.exp(-acc))
    sig = tl.where(mask_n[None, :], sig, 0.0)

    partial = tl.sum(sig, axis=1)

    tl.atomic_add(out_ptr + offs_m, partial, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Pre-transpose weight to (K, N) so inner loop loads contiguously along N
        with torch.no_grad():
            w_t = self.linear.weight.detach().t().contiguous()
        self.register_buffer('weight_t', w_t)

    def forward(self, x):
        x = x.contiguous().cuda()
        M, K = x.shape
        N = self.hidden_size
        W = self.weight_t  # (K, N)
        if W.device != x.device:
            W = W.to(x.device)
            self.weight_t = W
        B = self.linear.bias.contiguous()
        if B.device != x.device:
            B = B.to(x.device)

        out = torch.zeros((M,), device=x.device, dtype=torch.float32)

        def grid(meta):
            return (triton.cdiv(N, meta['BLOCK_N']), triton.cdiv(M, meta['BLOCK_M']))

        _fused_gemm_sigmoid_sum_kernel[grid](
            x, W, B, out,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
        )

        # Defensive: ensure out is correct (reset_to_zero handles autotune trials)
        return out.view(M, 1)