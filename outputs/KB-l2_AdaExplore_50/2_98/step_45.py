import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_kernel(
    X_ptr, Wt_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
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

    mask_m = offs_am < M
    mask_n = offs_bn < N

    x_ptrs = X_ptr + (offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    # Wt is (K, N): row k, col n -> contiguous in N
    w_ptrs = Wt_ptr + (offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remaining), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < k_remaining) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add bias
    offs_n_full = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    b_vals = tl.load(B_ptr + offs_n_full, mask=offs_n_full < N, other=0.0)
    acc += b_vals[None, :]

    offs_m_full = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    y_ptrs = Y_ptr + offs_m_full[:, None] * stride_ym + offs_n_full[None, :] * stride_yn
    mask = (offs_m_full[:, None] < M) & (offs_n_full[None, :] < N)
    tl.store(y_ptrs, acc, mask=mask)


@triton.jit
def pool_gelu_scale_max_kernel(
    Y_ptr, Out_ptr,
    M, N, P,
    SCALE: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # one program per row; reduce N -> P -> max
    pid = tl.program_id(0)
    row_ptr = Y_ptr + pid * N

    # We process the full row in chunks of (BLOCK_P * POOL) elements.
    # BLOCK_P = number of pooled outputs per chunk
    # chunk size in N = BLOCK_P * POOL
    inv_sqrt2 = 0.7071067811865475

    max_val = -float('inf')
    num_chunks = tl.cdiv(P, BLOCK_P)
    for c in range(0, num_chunks):
        p_offs = c * BLOCK_P + tl.arange(0, BLOCK_P)  # (BLOCK_P,)
        # for each pooled bin, sum POOL consecutive elements starting at p*POOL
        # Build 2D: (BLOCK_P, POOL)
        n_offs = p_offs[:, None] * POOL + tl.arange(0, POOL)[None, :]
        mask = (p_offs[:, None] < P)
        vals = tl.load(row_ptr + n_offs, mask=mask, other=0.0)
        pooled = tl.sum(vals, axis=1) / POOL  # (BLOCK_P,)
        # GELU
        gelu = 0.5 * pooled * (1.0 + tl.math.erf(pooled * inv_sqrt2))
        scaled = gelu * SCALE
        # mask invalid pooled entries to -inf
        scaled = tl.where(p_offs < P, scaled, -float('inf'))
        chunk_max = tl.max(scaled, axis=0)
        max_val = tl.maximum(max_val, chunk_max)

    tl.store(Out_ptr + pid, max_val)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = float(scale_factor)

        lin = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(lin.weight.detach().clone())
        self.bias = nn.Parameter(lin.bias.detach().clone())
        # Pre-transposed weight (K, N) for contiguous N-axis loads in GEMM
        self.register_buffer('weight_t', lin.weight.detach().clone().t().contiguous())

        assert out_features % pool_kernel_size == 0, "out_features must be divisible by pool_kernel_size"
        self.pooled_size = out_features // pool_kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        Wt = self.weight_t
        if Wt.device != x.device:
            Wt = Wt.to(x.device)
            self.weight_t = Wt
        # Keep weight_t in sync if weight has been updated (e.g. after training step)
        # In inference path we assume weight_t is current.
        B = self.bias.contiguous()

        M, K = x.shape
        N = self.out_features
        POOL = self.pool_kernel_size
        P = self.pooled_size

        Y = torch.empty((M, N), device=x.device, dtype=torch.float32)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
        gemm_bias_kernel[grid](
            x, Wt, B, Y,
            M, N, K,
            x.stride(0), x.stride(1),
            Wt.stride(0), Wt.stride(1),
            Y.stride(0), Y.stride(1),
        )

        out = torch.empty((M,), device=x.device, dtype=torch.float32)

        # P=512: use BLOCK_P=256 for fewer chunks
        BLOCK_P = 256
        if P < BLOCK_P:
            BLOCK_P = max(1, 1 << (P - 1).bit_length())

        pool_gelu_scale_max_kernel[(M,)](
            Y, out,
            M, N, P,
            SCALE=self.scale_factor,
            POOL=POOL,
            BLOCK_P=BLOCK_P,
            num_warps=4,
            num_stages=2,
        )

        return out