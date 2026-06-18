import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 16}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def splitk_gemm_sigmoid_rowsum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # K range for this split
    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k)[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + (k_start + offs_k)[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    k = k_start
    while k < k_end:
        k_remaining = k_end - k
        mask_k = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
        k += BLOCK_K

    # Only the last split_k program adds bias and applies sigmoid.
    # But we need full accumulation across splits before sigmoid.
    # So we atomically accumulate the dot, and a separate kernel does sigmoid+sum.
    # However that requires another full pass. Instead: we apply sigmoid only if SPLIT_K==1.
    # For SPLIT_K>1, we atomically add to a [M,N] buffer, then a separate kernel applies sigmoid+rowsum.
    # That's expensive. Better: use atomic-add of partial sigmoid is incorrect.
    #
    # Strategy here: write partial sums (without sigmoid) atomically; second kernel handles
    # sigmoid+rowsum. But that needs [M,N] buffer = 16GB. No.
    #
    # Use different approach: don't split K. SPLIT_K=1 always.
    pass


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 128}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_sigmoid_rowsum_kernel(
    x_ptr, w_ptr, b_ptr, partial_ptr,
    M, N, K, NTiles,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_pm, stride_pn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        mask_k = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

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
        # Pre-move to CUDA contiguous
        self._W = self.linear.weight.detach().contiguous().cuda()
        self._B = self.linear.bias.detach().contiguous().cuda()

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self._W
        B = self._B

        M, K = x.shape
        N = W.shape[0]

        BLOCK_M = 128 if M >= 128 else (64 if M >= 64 else (32 if M >= 32 else 16))
        # use power-of-2 BLOCK_M >= M for our M=128 case
        BLOCK_M = 128

        # Allocate partial sums with safe upper bound (min BLOCK_N = 64)
        max_n_tiles = (N + 64 - 1) // 64
        partial = torch.empty((M, max_n_tiles), device=x.device, dtype=torch.float32)

        def grid(meta):
            n_tiles = triton.cdiv(N, meta['BLOCK_N'])
            return (triton.cdiv(M, BLOCK_M), n_tiles)

        fused_gemm_sigmoid_rowsum_kernel[grid](
            x, W, B, partial,
            M, N, K, max_n_tiles,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            partial.stride(0), partial.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_K=32,
        )

        best_config = fused_gemm_sigmoid_rowsum_kernel.best_config
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