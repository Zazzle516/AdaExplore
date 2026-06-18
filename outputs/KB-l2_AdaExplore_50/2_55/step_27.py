import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_pool_sum_kernel(
    x_ptr,        # [M, K]
    wt_ptr,       # [K, N]  (W transposed at init)
    b_ptr,        # [N]
    out_ptr,      # [M]
    M, N, K,
    KERNEL_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    N_SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)  # n-tile group index

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M

    # N tiles per program
    n_tiles = tl.cdiv(N, BLOCK_N)
    tiles_per_split = tl.cdiv(n_tiles, N_SPLITS)
    n_tile_start = pid_n * tiles_per_split
    n_tile_end = tl.minimum(n_tile_start + tiles_per_split, n_tiles)

    # scalar accumulator per row of the M-block
    row_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    x_base = x_ptr + offs_m[:, None] * K  # [BLOCK_M, 1]

    for nt in range(n_tile_start, n_tile_end):
        n_off = nt * BLOCK_N
        offs_n = n_off + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        # GEMM accumulator [BLOCK_M, BLOCK_N]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k_idx = k_start + offs_k
            k_mask = k_idx < K

            x_ptrs = x_base + k_idx[None, :]
            x_vals = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

            w_ptrs = wt_ptr + k_idx[:, None] * N + offs_n[None, :]
            w_vals = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

            acc += tl.dot(x_vals, w_vals, allow_tf32=True)

        # add bias
        b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
        acc = acc + b_vals[None, :]

        # set masked-out positions to -inf so they don't affect max
        acc = tl.where(n_mask[None, :], acc, -float('inf'))

        # reshape [BLOCK_M, BLOCK_N/KS, KS], max over KS
        acc3d = tl.reshape(acc, (BLOCK_M, BLOCK_N // KERNEL_SIZE, KERNEL_SIZE))
        pooled = tl.max(acc3d, axis=2)  # [BLOCK_M, BLOCK_N/KS]
        row_acc += tl.sum(pooled, axis=1)

    if N_SPLITS == 1:
        row_acc = row_acc * SCALE
        tl.store(out_ptr + offs_m, row_acc, mask=m_mask)
    else:
        # atomic add per row
        tl.atomic_add(out_ptr + offs_m, row_acc, mask=m_mask)


@triton.jit
def scale_kernel(out_ptr, M, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    v = tl.load(out_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, v * SCALE, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = float(scale_factor)
        self.matmul = nn.Linear(in_features, out_features)

        # Pre-store W transposed for coalesced K-major loads
        with torch.no_grad():
            wt = self.matmul.weight.detach().t().contiguous()
        self.register_buffer('wt', wt, persistent=False)

        # tune N_SPLITS for parallelism. B=128, SMs=128. With BLOCK_M=64 -> 2 m-tiles
        # giving 2*N_SPLITS programs. Use N_SPLITS=64 -> 128 programs.
        self.N_SPLITS = 64

    def forward(self, x):
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()
        device = x.device

        if self.wt.device != device:
            self.wt = self.wt.to(device)
        wt = self.wt
        b = self.matmul.bias.to(device).contiguous()

        M = x.shape[0]
        K = self.in_features
        N = self.out_features
        N_SPLITS = self.N_SPLITS

        if N_SPLITS == 1:
            out = torch.empty(M, device=device, dtype=torch.float32)
        else:
            out = torch.zeros(M, device=device, dtype=torch.float32)

        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), N_SPLITS)

        fused_gemm_pool_sum_kernel[grid](
            x, wt, b, out,
            M, N, K,
            KERNEL_SIZE=self.kernel_size,
            SCALE=self.scale_factor,
            N_SPLITS=N_SPLITS,
        )

        if N_SPLITS > 1:
            BLOCK = 128
            grid2 = ((M + BLOCK - 1) // BLOCK,)
            scale_kernel[grid2](out, M, SCALE=self.scale_factor, BLOCK=BLOCK)

        return out