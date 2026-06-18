import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_sigmoid_sum_kernel(
    x_ptr, wt_ptr, b_ptr, partial_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wtk, stride_wtn,
    stride_pm, stride_pn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # wt is [K, N], contiguous on N (stride_wtn = 1)
    wt_ptrs = wt_ptr + offs_k[:, None] * stride_wtk + offs_n[None, :] * stride_wtn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        mask_k = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(wt_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        wt_ptrs += BLOCK_K * stride_wtk

    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]

    sig = tl.sigmoid(acc)
    sig = tl.where(mask_n[None, :], sig, 0.0)

    partial = tl.sum(sig, axis=1)  # [BLOCK_M]

    p_ptrs = partial_ptr + offs_m * stride_pm + pid_n * stride_pn
    tl.store(p_ptrs, partial, mask=mask_m)


@triton.jit
def reduce_partials_kernel(
    partial_ptr, out_ptr,
    M, NTiles,
    stride_pm, stride_pn,
    BLOCK_T: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_T)
    mask = offs < NTiles
    vals = tl.load(partial_ptr + pid_m * stride_pm + offs * stride_pn, mask=mask, other=0.0)
    s = tl.sum(vals, axis=0)
    tl.store(out_ptr + pid_m, s)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Pre-transpose W to [K, N] contiguous on N axis
        with torch.no_grad():
            wt = self.linear.weight.detach().t().contiguous().cuda()  # [K, N]
            b = self.linear.bias.detach().contiguous().cuda()         # [N]
        self.register_buffer('wt', wt)
        self.register_buffer('bias_cached', b)

    def forward(self, x):
        x = x.contiguous().cuda()
        wt = self.wt
        B = self.bias_cached

        M, K = x.shape
        N = wt.shape[1]

        # Upper bound on n_tiles (smallest BLOCK_N in configs is 128)
        max_n_tiles = (N + 128 - 1) // 128
        partial = torch.empty((M, max_n_tiles), device=x.device, dtype=torch.float32)

        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        fused_linear_sigmoid_sum_kernel[grid](
            x, wt, B, partial,
            M, N, K,
            x.stride(0), x.stride(1),
            wt.stride(0), wt.stride(1),
            partial.stride(0), partial.stride(1),
        )

        best_config = fused_linear_sigmoid_sum_kernel.best_config
        block_n = best_config.kwargs['BLOCK_N']
        n_tiles = (N + block_n - 1) // block_n

        out = torch.empty((M, 1), device=x.device, dtype=torch.float32)

        BLOCK_T = 1
        while BLOCK_T < n_tiles:
            BLOCK_T *= 2
        BLOCK_T = max(BLOCK_T, 16)

        reduce_partials_kernel[(M,)](
            partial, out,
            M, n_tiles,
            partial.stride(0), partial.stride(1),
            BLOCK_T=BLOCK_T,
        )

        return out