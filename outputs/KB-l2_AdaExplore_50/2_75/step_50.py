import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(
    A, B, C, bias,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias_vals = tl.load(bias + offs_n, mask=mask_n, other=0.0)
    acc += bias_vals[None, :]

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


# GroupNorm + per-row min. One program per (row, group_tile). For C_per_g=16
# and num_groups=512, we put GROUPS_PER_PROG groups per program to amortize.
@triton.jit
def gn_min_kernel(
    X_ptr, gamma_ptr, beta_ptr, OUT_ptr,
    M, N, eps,
    C_PER_G: tl.constexpr,
    GROUPS_PER_PROG: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    row = tl.program_id(0)
    gtile = tl.program_id(1)

    g_start = gtile * GROUPS_PER_PROG
    # Process GROUPS_PER_PROG groups, each of size C_PER_G, as a 2D tile
    # shape: [GROUPS_PER_PROG, C_PER_G]
    offs_g = g_start + tl.arange(0, GROUPS_PER_PROG)  # group indices
    offs_c = tl.arange(0, C_PER_G)                    # within group

    # full channel offsets
    chan = offs_g[:, None] * C_PER_G + offs_c[None, :]  # [GPP, CPG]
    x_ptrs = X_ptr + row * N + chan
    x = tl.load(x_ptrs)  # no mask needed since N is divisible

    # per-group mean/var
    sum_x = tl.sum(x, axis=1)  # [GPP]
    mean = sum_x / C_PER_G
    diff = x - mean[:, None]
    var = tl.sum(diff * diff, axis=1) / C_PER_G
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(gamma_ptr + chan)
    beta = tl.load(beta_ptr + chan)

    y = diff * rstd[:, None] * gamma + beta  # [GPP, CPG]

    # min over this tile
    tile_min = tl.min(tl.min(y, axis=1), axis=0)

    tl.atomic_min(OUT_ptr + row, tile_min)


# 2D-tiled bias add: out[c, m] = row_min[m] + bias[c]
@triton.jit
def bias_add_kernel(
    min_ptr, bias_ptr, out_ptr,
    M, C,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    mask_c = offs_c < C
    mask_m = offs_m < M

    m_vals = tl.load(min_ptr + offs_m, mask=mask_m, other=0.0)
    b_vals = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)

    out = m_vals[None, :] + b_vals[:, None]  # [BLOCK_C, BLOCK_M]

    # Output layout (1, C, M, 1) -> flat index c*M + m
    out_ptrs = out_ptr + offs_c[:, None] * M + offs_m[None, :]
    tl.store(out_ptrs, out, mask=mask_c[:, None] & mask_m[None, :])


def triton_gemm(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    gemm_kernel[grid](
        x, weight, out, bias,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        out.stride(0), out.stride(1),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups

    def forward(self, x):
        x = x.contiguous()
        y = triton_gemm(x, self.gemm.weight, self.gemm.bias)

        M, N = y.shape
        C_per_g = N // self.num_groups  # 16
        num_groups = self.num_groups

        # Choose groups per program
        if C_per_g <= 16:
            GROUPS_PER_PROG = 16
        elif C_per_g <= 32:
            GROUPS_PER_PROG = 8
        else:
            GROUPS_PER_PROG = 4
        # ensure divides num_groups
        while num_groups % GROUPS_PER_PROG != 0:
            GROUPS_PER_PROG //= 2
        num_tiles = num_groups // GROUPS_PER_PROG

        min_buf = torch.full((M,), float('inf'), device=y.device, dtype=y.dtype)

        gn_min_kernel[(M, num_tiles)](
            y, self.group_norm.weight, self.group_norm.bias, min_buf,
            M, N, self.group_norm.eps,
            C_PER_G=C_per_g,
            GROUPS_PER_PROG=GROUPS_PER_PROG,
            NUM_GROUPS=num_groups,
            num_warps=4,
        )

        out_features = self.out_features
        out = torch.empty((1, out_features, M, 1), device=y.device, dtype=y.dtype)
        bias_flat = self.bias.view(-1).contiguous()

        BLOCK_M = 128
        BLOCK_C = 64
        grid = (triton.cdiv(out_features, BLOCK_C), triton.cdiv(M, BLOCK_M))
        bias_add_kernel[grid](
            min_buf, bias_flat, out,
            M, out_features,
            BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C,
            num_warps=4,
        )
        return out