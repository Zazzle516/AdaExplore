import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def stage1_gemm_partial_kernel(
    x_ptr, w_ptr, b_ptr,
    max_ptr, sum_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_mm, stride_mn,
    NUM_N_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x_base = x_ptr + offs_m[:, None] * stride_xm
    w_base = w_ptr + offs_n[None, :] * stride_wn

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        x_ptrs = x_base + k_offs[None, :] * stride_xk
        w_ptrs = w_base + k_offs[:, None] * stride_wk
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x, w)

    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]
    n_mask = offs_n[None, :] < N
    acc = tl.where(n_mask, acc, -float('inf'))

    tile_max = tl.max(acc, axis=1)
    e = tl.exp(acc - tile_max[:, None])
    e = tl.where(n_mask, e, 0.0)
    tile_sum = tl.sum(e, axis=1)

    m_mask = offs_m < M
    out_max = max_ptr + offs_m * stride_mm + pid_n * stride_mn
    out_sum = sum_ptr + offs_m * stride_mm + pid_n * stride_mn
    tl.store(out_max, tile_max, mask=m_mask)
    tl.store(out_sum, tile_sum, mask=m_mask)


@triton.jit
def stage2_combine_kernel(
    max_ptr, sum_ptr, out_ptr,
    M, NUM_TILES,
    stride_mm, stride_mn,
    BLOCK_M: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    offs_t = tl.arange(0, BLOCK_T)
    t_mask = offs_t < NUM_TILES

    max_ptrs = max_ptr + offs_m[:, None] * stride_mm + offs_t[None, :] * stride_mn
    sum_ptrs = sum_ptr + offs_m[:, None] * stride_mm + offs_t[None, :] * stride_mn

    mask2d = m_mask[:, None] & t_mask[None, :]
    tile_max = tl.load(max_ptrs, mask=mask2d, other=-float('inf'))
    tile_sum = tl.load(sum_ptrs, mask=mask2d, other=0.0)

    row_max = tl.max(tile_max, axis=1)
    # weight: exp(tile_max - row_max) * tile_sum
    weighted = tl.exp(tile_max - row_max[:, None]) * tile_sum
    weighted = tl.where(mask2d, weighted, 0.0)
    row_sum = tl.sum(weighted, axis=1)

    x = row_max + tl.log(row_sum)
    # LeakyReLU twice
    x = tl.where(x >= 0, x, x * 0.01)
    x = tl.where(x >= 0, x, x * 0.01)
    # GELU twice
    inv_sqrt2 = 0.7071067811865475
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(out_ptr + offs_m, x, mask=m_mask)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        with torch.no_grad():
            wt = self.linear.weight.detach().t().contiguous()  # (K, N)
        self.register_buffer('w_kn', wt)
        if bias:
            self.register_buffer('b_buf', self.linear.bias.detach().contiguous())
        else:
            self.register_buffer('b_buf', torch.zeros(out_features))

    def forward(self, x):
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()
        W = self.w_kn
        if W.device != x.device:
            W = W.to(x.device)
            self.w_kn = W
        b = self.b_buf
        if b.device != x.device:
            b = b.to(x.device)
            self.b_buf = b

        M, K = x.shape
        N = W.shape[1]

        # Allocate partials with worst-case BLOCK_N tiles; we use a fixed shape
        # by binding NUM_N_TILES via grid (and using runtime cdiv inside kernel)
        # Simpler approach: do two launches, get num_n_tiles from autotuner meta.
        partial_max = torch.empty((M, 64), device=x.device, dtype=torch.float32)
        partial_sum = torch.empty((M, 64), device=x.device, dtype=torch.float32)

        def grid(meta):
            num_n_tiles = triton.cdiv(N, meta['BLOCK_N'])
            return (triton.cdiv(M, meta['BLOCK_M']), num_n_tiles)

        # We need partial buffer sized to num_n_tiles; allocate inside via lambda hack
        # Use a callable that allocates after autotune picks config.
        # Workaround: allocate generously and ignore unused tail.
        # Max needed = cdiv(N, min BLOCK_N=64) = N/64
        max_tiles = (N + 63) // 64
        if partial_max.shape[1] < max_tiles:
            partial_max = torch.empty((M, max_tiles), device=x.device, dtype=torch.float32)
            partial_sum = torch.empty((M, max_tiles), device=x.device, dtype=torch.float32)

        stage1_gemm_partial_kernel[grid](
            x, W, b,
            partial_max, partial_sum,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            partial_max.stride(0), partial_max.stride(1),
            NUM_N_TILES=max_tiles,
        )

        # Determine actual num tiles used from autotune
        cfg = stage1_gemm_partial_kernel.best_config
        block_n = cfg.kwargs['BLOCK_N']
        num_n_tiles = (N + block_n - 1) // block_n

        out = torch.empty(M, device=x.device, dtype=torch.float32)

        BLOCK_M2 = 64
        BLOCK_T = _next_pow2(num_n_tiles)
        if BLOCK_T < 8:
            BLOCK_T = 8

        grid2 = (triton.cdiv(M, BLOCK_M2),)
        stage2_combine_kernel[grid2](
            partial_max, partial_sum, out,
            M, num_n_tiles,
            partial_max.stride(0), partial_max.stride(1),
            BLOCK_M=BLOCK_M2,
            BLOCK_T=BLOCK_T,
        )

        return out.view(M, 1)