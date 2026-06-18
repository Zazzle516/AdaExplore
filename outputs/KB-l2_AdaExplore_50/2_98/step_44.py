import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def row_max_kernel(
    Pmax_ptr, Out_ptr,
    M, NTILES,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    offs = tl.arange(0, BLOCK_T)
    mask = offs < NTILES
    NEG_INF = float('-inf')
    vals = tl.load(Pmax_ptr + pid * NTILES + offs, mask=mask, other=NEG_INF)
    m = tl.max(vals, axis=0)
    tl.store(Out_ptr + pid, m)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_kernel_size = int(pool_kernel_size)
        self.scale_factor = float(scale_factor)

        lin = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(lin.weight.detach().clone())
        self.bias = nn.Parameter(lin.bias.detach().clone())
        # Pre-transposed weight: (K, N) contiguous, so N is contiguous axis for GEMM B-load.
        self.register_buffer('weight_t', lin.weight.detach().clone().t().contiguous())

        assert out_features % self.pool_kernel_size == 0
        self.pooled_size = out_features // self.pool_kernel_size

    def forward(self, x):
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()
        # Keep weight_t in sync with weight (in case weight was modified)
        Wt = self.weight_t
        B = self.bias

        M, K = x.shape
        N = self.out_features
        POOL = self.pool_kernel_size

        BLOCK_N = 128
        NTILES = (N + BLOCK_N - 1) // BLOCK_N

        partial_max = torch.empty((M, NTILES), device=x.device, dtype=torch.float32)

        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']) * NTILES,)

        _fused_kernel_static[grid](
            x, Wt, B, partial_max,
            M, N, K, NTILES,
            x.stride(0), x.stride(1),
            Wt.stride(0), Wt.stride(1),
            SCALE=self.scale_factor,
            POOL=POOL,
            BLOCK_N=BLOCK_N,
        )

        out = torch.empty((M,), device=x.device, dtype=torch.float32)
        BLOCK_T = 1
        while BLOCK_T < NTILES:
            BLOCK_T *= 2
        if BLOCK_T < 8:
            BLOCK_T = 8

        row_max_kernel[(M,)](
            partial_max, out,
            M, NTILES,
            BLOCK_T=BLOCK_T,
            num_warps=2,
            num_stages=2,
        )

        return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 32, 'GROUP_M': 4}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_kernel_static(
    X_ptr, Wt_ptr, B_ptr, PartialMax_ptr,
    M, N, K, NTILES,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    SCALE: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    am_mask = offs_am < M
    bn_mask = offs_bn < N

    a_rows = tl.where(am_mask, offs_am, 0)
    b_cols = tl.where(bn_mask, offs_bn, 0)

    x_ptrs = X_ptr + (a_rows[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    # Wt is (K, N) with stride_wk along K, stride_wn=1 along N (contiguous)
    w_ptrs = Wt_ptr + (offs_k[:, None] * stride_wk + b_cols[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    b_vals = tl.load(B_ptr + offs_bn, mask=bn_mask, other=0.0)
    acc += b_vals[None, :]

    BLOCK_P: tl.constexpr = BLOCK_N // POOL
    acc_3d = tl.reshape(acc, (BLOCK_M, BLOCK_P, POOL))
    pooled = tl.sum(acc_3d, axis=2) * (1.0 / POOL)

    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * pooled * (1.0 + tl.math.erf(pooled * inv_sqrt2))
    scaled = gelu * SCALE

    p_offs = pid_n * BLOCK_P + tl.arange(0, BLOCK_P)
    P_total = N // POOL
    p_valid = p_offs < P_total
    NEG_INF = float('-inf')
    scaled = tl.where(am_mask[:, None] & p_valid[None, :], scaled, NEG_INF)

    tile_max = tl.max(scaled, axis=1)

    out_row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_ptrs = PartialMax_ptr + out_row_idx * NTILES + pid_n
    tl.store(out_ptrs, tile_max, mask=am_mask)