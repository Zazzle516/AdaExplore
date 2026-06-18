import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# =========================================================
# Tiled GEMM with bias + Swish epilogue:  Y = swish(X @ W^T + b_linear) + bias_param
# =========================================================
GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_swish_bias_kernel(
    X_ptr, W_ptr, Bl_ptr, Bp_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w, allow_tf32=True)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # add linear bias
    bl = tl.load(Bl_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bl[None, :]

    # swish: x * sigmoid(x)
    acc = acc * tl.sigmoid(acc)

    # add bias param
    bp = tl.load(Bp_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bp[None, :]

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


# =========================================================
# GroupNorm kernel - one program per (row, group)
# =========================================================
@triton.jit
def group_norm_kernel(
    X_ptr, Y_ptr, Gamma_ptr, Beta_ptr,
    M, C, G, CPG,
    eps,
    BLOCK_CPG: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    total_groups = M * G
    group_start = pid * ROWS_PER_PROG

    offs = tl.arange(0, BLOCK_CPG)
    mask_c = offs < CPG

    for i in tl.static_range(ROWS_PER_PROG):
        gid = group_start + i
        valid = gid < total_groups
        row = gid // G
        grp = gid % G

        base = row * C + grp * CPG
        x = tl.load(X_ptr + base + offs, mask=mask_c & valid, other=0.0).to(tl.float32)

        cnt = CPG
        mean = tl.sum(x, axis=0) / cnt
        xc = tl.where(mask_c, x - mean, 0.0)
        var = tl.sum(xc * xc, axis=0) / cnt
        rstd = 1.0 / tl.sqrt(var + eps)

        g = tl.load(Gamma_ptr + grp * CPG + offs, mask=mask_c, other=0.0)
        b = tl.load(Beta_ptr + grp * CPG + offs, mask=mask_c, other=0.0)

        y = xc * rstd * g + b
        tl.store(Y_ptr + base + offs, y, mask=mask_c & valid)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups

        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        x = x.contiguous().cuda()
        M, K = x.shape
        N = self.out_features

        W = self.matmul.weight  # (N, K)
        bl = self.matmul.bias   # (N,)
        bp = self.bias           # (N,)

        y = torch.empty((M, N), device=x.device, dtype=x.dtype)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
        gemm_swish_bias_kernel[grid](
            x, W, bl, bp, y,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            y.stride(0), y.stride(1),
        )

        # GroupNorm
        CPG = N // self.num_groups
        BLOCK_CPG = _next_pow2(CPG)
        out = torch.empty_like(y)
        ROWS_PER_PROG = 4
        total = M * self.num_groups
        grid_gn = ((total + ROWS_PER_PROG - 1) // ROWS_PER_PROG,)
        group_norm_kernel[grid_gn](
            y, out, self.group_norm.weight, self.group_norm.bias,
            M, N, self.num_groups, CPG,
            float(self.group_norm.eps),
            BLOCK_CPG=BLOCK_CPG,
            ROWS_PER_PROG=ROWS_PER_PROG,
            num_warps=2,
            num_stages=2,
        )
        return out