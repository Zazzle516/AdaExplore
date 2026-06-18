import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_gemm_sigmoid_partial_kernel(
    x_ptr, w_ptr, b_ptr, partial_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,  # w is (K, N)
    stride_pm, stride_pn,  # partial is (M, num_n_tiles)
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

    p_ptrs = partial_ptr + offs_m * stride_pm + pid_n * stride_pn
    tl.store(p_ptrs, partial, mask=mask_m)


@triton.jit
def _reduce_partial_kernel(
    partial_ptr, out_ptr,
    M, NUM_TILES,
    stride_pm, stride_pn,
    BLOCK_T: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    offs_t = tl.arange(0, BLOCK_T)
    mask_t = offs_t < NUM_TILES
    p = tl.load(partial_ptr + pid_m * stride_pm + offs_t * stride_pn,
                mask=mask_t, other=0.0)
    s = tl.sum(p, axis=0)
    tl.store(out_ptr + pid_m, s)


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

        # We don't know the chosen BLOCK_N until autotune runs.
        # Allocate a worst-case partial buffer keyed on the smallest BLOCK_N we use (64).
        # Use a single large buffer; the kernel uses pid_n indexing with stride_pn.
        MAX_N_TILES = (N + 63) // 64  # worst case for smallest BLOCK_N=64
        partial = torch.empty((M, MAX_N_TILES), device=x.device, dtype=torch.float32)
        out = torch.empty((M,), device=x.device, dtype=torch.float32)

        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        _fused_gemm_sigmoid_partial_kernel[grid](
            x, W, B, partial,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            partial.stride(0), partial.stride(1),
        )

        # Determine actual num_n_tiles from the autotuned config
        best_config = _fused_gemm_sigmoid_partial_kernel.best_config
        BLOCK_N_chosen = best_config.kwargs['BLOCK_N']
        num_n_tiles = (N + BLOCK_N_chosen - 1) // BLOCK_N_chosen

        # Reduce partial along n-tile axis
        # Pick BLOCK_T as next pow2 >= num_n_tiles
        BLOCK_T = 1
        while BLOCK_T < num_n_tiles:
            BLOCK_T *= 2
        BLOCK_T = max(BLOCK_T, 16)

        _reduce_partial_kernel[(M,)](
            partial, out,
            M, num_n_tiles,
            partial.stride(0), partial.stride(1),
            BLOCK_T=BLOCK_T,
        )

        return out.view(M, 1)