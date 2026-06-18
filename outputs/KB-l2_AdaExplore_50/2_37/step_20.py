import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_swish_bias_partial_kernel(
    x_ptr, w_ptr, lb_ptr, bias_ptr,
    y_ptr, sum_ptr, sumsq_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    stride_sm, stride_sg,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_TILE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_mask = offs_k[None, :] < (K - k)
        x = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask, other=0.0)
        w = tl.load(w_ptrs, mask=mask_n[:, None] & k_mask, other=0.0)
        acc += tl.dot(x, tl.trans(w), allow_tf32=True)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # add linear bias
    lb = tl.load(lb_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + lb[None, :]
    # swish
    acc = acc * tl.sigmoid(acc)
    # add extra bias
    b = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]

    # store Y
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

    # partial sums per group within this tile
    # BLOCK_N must be GROUPS_PER_TILE * GROUP_SIZE
    acc_r = tl.reshape(acc, (BLOCK_M, GROUPS_PER_TILE, GROUP_SIZE))
    s = tl.sum(acc_r, axis=2)
    sq = tl.sum(acc_r * acc_r, axis=2)

    # group indices for this tile
    g_start = pid_n * GROUPS_PER_TILE
    offs_g = g_start + tl.arange(0, GROUPS_PER_TILE)

    s_ptrs = sum_ptr + offs_m[:, None] * stride_sm + offs_g[None, :] * stride_sg
    sq_ptrs = sumsq_ptr + offs_m[:, None] * stride_sm + offs_g[None, :] * stride_sg
    tl.store(s_ptrs, s, mask=mask_m[:, None])
    tl.store(sq_ptrs, sq, mask=mask_m[:, None])


@triton.jit
def groupnorm_apply_kernel(
    y_ptr, sum_ptr, sumsq_ptr, gamma_ptr, beta_ptr, out_ptr,
    M, N, G,
    stride_ym, stride_yn,
    stride_sm, stride_sg,
    eps,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # group index for each n
    g_idx = offs_n // GROUP_SIZE

    s = tl.load(sum_ptr + pid_m * stride_sm + g_idx * stride_sg, mask=mask_n, other=0.0)
    sq = tl.load(sumsq_ptr + pid_m * stride_sm + g_idx * stride_sg, mask=mask_n, other=0.0)

    inv_gs = 1.0 / GROUP_SIZE
    mean = s * inv_gs
    var = sq * inv_gs - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    y = tl.load(y_ptr + pid_m * stride_ym + offs_n * stride_yn, mask=mask_n, other=0.0)
    g = tl.load(gamma_ptr + offs_n, mask=mask_n, other=1.0)
    b = tl.load(beta_ptr + offs_n, mask=mask_n, other=0.0)

    out = (y - mean) * rstd * g + b
    tl.store(out_ptr + pid_m * stride_ym + offs_n * stride_yn, out, mask=mask_n)


@triton.jit
def reduce_groups_kernel(
    sum_ptr, sumsq_ptr,
    M, G, NUM_TILES,
    BLOCK_TILES: tl.constexpr,
):
    # not used in this design; left as placeholder
    pass


def fused_forward(x, weight, lin_bias, bias, gamma, beta, num_groups, eps):
    M, K = x.shape
    N, _ = weight.shape
    G = num_groups
    GROUP_SIZE = N // G

    BLOCK_M = 64
    BLOCK_N = 128
    assert BLOCK_N % GROUP_SIZE == 0
    GROUPS_PER_TILE = BLOCK_N // GROUP_SIZE

    y = torch.empty((M, N), device=x.device, dtype=torch.float32)
    # per-row, per-group partial sums (one tile per group region since BLOCK_N covers GROUPS_PER_TILE full groups)
    sum_buf = torch.empty((M, G), device=x.device, dtype=torch.float32)
    sumsq_buf = torch.empty((M, G), device=x.device, dtype=torch.float32)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    fused_gemm_swish_bias_partial_kernel[grid](
        x, weight, lin_bias, bias,
        y, sum_buf, sumsq_buf,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        y.stride(0), y.stride(1),
        sum_buf.stride(0), sum_buf.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        GROUP_SIZE=GROUP_SIZE,
        GROUPS_PER_TILE=GROUPS_PER_TILE,
    )

    out = torch.empty_like(y)
    BLOCK_N2 = 256
    grid2 = (M, triton.cdiv(N, BLOCK_N2))
    groupnorm_apply_kernel[grid2](
        y, sum_buf, sumsq_buf, gamma, beta, out,
        M, N, G,
        y.stride(0), y.stride(1),
        sum_buf.stride(0), sum_buf.stride(1),
        eps,
        GROUP_SIZE=GROUP_SIZE,
        BLOCK_N=BLOCK_N2,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.num_groups = num_groups
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.matmul.weight.contiguous()
        lb = self.matmul.bias.contiguous()
        b = self.bias.contiguous()
        g = self.group_norm.weight.contiguous()
        bt = self.group_norm.bias.contiguous()
        eps = self.group_norm.eps
        return fused_forward(x, w, lb, b, g, bt, self.num_groups, eps)