import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_sigmoid_sum_kernel(
    x_ptr, w_ptr, b_ptr, partial_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_pm, stride_pn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    b = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + b[None, :]
    sig = tl.sigmoid(acc)
    sig = tl.where(m_mask[:, None] & n_mask[None, :], sig, 0.0)

    row_sum = tl.sum(sig, axis=1)

    p_ptrs = partial_ptr + offs_m * stride_pm + pid_n * stride_pn
    tl.store(p_ptrs, row_sum, mask=m_mask)


@triton.jit
def reduce_partials_kernel(
    partial_ptr, out_ptr,
    M, NUM_N_TILES,
    stride_pm, stride_pn,
    BLOCK_T: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_t = tl.arange(0, BLOCK_T)
    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)
    for t_start in range(0, NUM_N_TILES, BLOCK_T):
        idx = t_start + offs_t
        mask = idx < NUM_N_TILES
        ptrs = partial_ptr + pid_m * stride_pm + idx * stride_pn
        v = tl.load(ptrs, mask=mask, other=0.0)
        acc += v
    s = tl.sum(acc, axis=0)
    tl.store(out_ptr + pid_m, s)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        with torch.no_grad():
            self._wT = self.linear.weight.detach().t().contiguous().cuda()
            self._b = self.linear.bias.detach().contiguous().cuda()

    def forward(self, x):
        x = x.contiguous().cuda()
        if self._wT.device != x.device:
            self._wT = self._wT.to(x.device)
            self._b = self._b.to(x.device)

        M, K = x.shape
        N = self.hidden_size

        out = torch.empty((M,), device=x.device, dtype=torch.float32)

        # Allocate worst-case partials (smallest BLOCK_N in configs = 64)
        max_num_n_tiles = triton.cdiv(N, 64)
        partials = torch.empty((M, max_num_n_tiles), device=x.device, dtype=torch.float32)

        def grid(meta):
            return (
                triton.cdiv(M, meta['BLOCK_M']),
                triton.cdiv(N, meta['BLOCK_N']),
            )

        fused_gemm_sigmoid_sum_kernel[grid](
            x, self._wT, self._b, partials,
            M, N, K,
            x.stride(0), x.stride(1),
            self._wT.stride(0), self._wT.stride(1),
            partials.stride(0), partials.stride(1),
        )

        best_cfg = fused_gemm_sigmoid_sum_kernel.best_config
        actual_block_n = best_cfg.kwargs['BLOCK_N']
        num_n_tiles = triton.cdiv(N, actual_block_n)

        BLOCK_T = 256
        reduce_partials_kernel[(M,)](
            partials, out,
            M, num_n_tiles,
            partials.stride(0), partials.stride(1),
            BLOCK_T=BLOCK_T,
        )
        return out.view(M, 1)