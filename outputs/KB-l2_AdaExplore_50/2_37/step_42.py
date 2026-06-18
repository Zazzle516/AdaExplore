import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
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
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x = tl.load(x_ptrs)
        w = tl.load(w_ptrs)
        acc += tl.dot(x, w, allow_tf32=True)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    bl = tl.load(Bl_ptr + offs_n)
    acc = acc + bl[None, :]
    acc = acc * tl.sigmoid(acc)
    bp = tl.load(Bp_ptr + offs_n)
    acc = acc + bp[None, :]

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc)


@triton.jit
def group_norm_kernel(
    X_ptr, Y_ptr, Gamma_ptr, Beta_ptr,
    M, C, G, CPG,
    eps,
    GROUPS_PER_PROG: tl.constexpr,
    BLOCK_CPG: tl.constexpr,
):
    pid = tl.program_id(0)
    groups_per_row = G // GROUPS_PER_PROG
    row = pid // groups_per_row
    grp_block = pid % groups_per_row
    grp_start = grp_block * GROUPS_PER_PROG

    offs_c = tl.arange(0, BLOCK_CPG)
    offs_g = tl.arange(0, GROUPS_PER_PROG)
    mask = offs_c[None, :] < CPG

    base = row * C + grp_start * CPG
    # shape: [GROUPS_PER_PROG, BLOCK_CPG]
    ptrs = X_ptr + base + offs_g[:, None] * CPG + offs_c[None, :]
    x = tl.load(ptrs, mask=mask, other=0.0)

    cnt = CPG.to(tl.float32)
    sum_x = tl.sum(x, axis=1)
    mean = sum_x / cnt
    xc = tl.where(mask, x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, axis=1) / cnt
    rstd = tl.rsqrt(var + eps)

    g_ptrs = Gamma_ptr + grp_start * CPG + offs_g[:, None] * CPG + offs_c[None, :]
    b_ptrs = Beta_ptr + grp_start * CPG + offs_g[:, None] * CPG + offs_c[None, :]
    g = tl.load(g_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)

    y = xc * rstd[:, None] * g + b
    tl.store(Y_ptr + base + offs_g[:, None] * CPG + offs_c[None, :], y, mask=mask)


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

        W = self.matmul.weight
        bl = self.matmul.bias
        bp = self.bias

        y = torch.empty((M, N), device=x.device, dtype=x.dtype)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
        gemm_swish_bias_kernel[grid](
            x, W, bl, bp, y,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            y.stride(0), y.stride(1),
        )

        CPG = N // self.num_groups
        BLOCK_CPG = _next_pow2(CPG)
        # pick groups per program so total programs = M * G / GPP
        GPP = 8
        while self.num_groups % GPP != 0:
            GPP //= 2
        if GPP < 1:
            GPP = 1
        out = torch.empty_like(y)
        group_norm_kernel[(M * (self.num_groups // GPP),)](
            y, out, self.group_norm.weight, self.group_norm.bias,
            M, N, self.num_groups, CPG,
            float(self.group_norm.eps),
            GROUPS_PER_PROG=GPP,
            BLOCK_CPG=BLOCK_CPG,
            num_warps=2,
        )
        return out