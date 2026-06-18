import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


# ---------------- GEMM + bias + hardtanh + mish ----------------
# BLOCK_N must be a multiple of CG (=32). Choose BLOCK_N = 256 -> 8 groups per tile.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_act_kernel(
    A_ptr, B_ptr, BIAS_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
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

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        mask_k = offs_k < k_remaining
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # bias
    bias = tl.load(BIAS_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # hardtanh
    acc = tl.minimum(tl.maximum(acc, -1.0), 1.0)

    # mish: x * tanh(softplus(x))
    sp = tl.log(1.0 + tl.exp(acc))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    acc = acc * th

    # Store as fp16 to halve bandwidth feeding GroupNorm
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])


def gemm_bias_act(x, W, bias_lin, bias_extra):
    M, K = x.shape
    N = W.shape[0]
    bias = (bias_lin + bias_extra).contiguous()
    x = x.contiguous()
    Wt = W.t().contiguous()
    out = torch.empty((M, N), device=x.device, dtype=torch.float16)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_bias_act_kernel[grid](
        x, Wt, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        Wt.stride(0), Wt.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


# ---------------- GroupNorm: persistent over rows, multiple groups per program ----------------

@triton.jit
def group_norm_kernel_packed(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, C, G, CG,
    eps,
    GROUPS_PER_PROG: tl.constexpr,
    CG_C: tl.constexpr,
):
    pid = tl.program_id(0)
    # number of program groups = (M * G) / GROUPS_PER_PROG
    # Each program handles GROUPS_PER_PROG consecutive groups within one row.
    groups_per_row = G
    progs_per_row = groups_per_row // GROUPS_PER_PROG
    row = pid // progs_per_row
    grp_start = (pid % progs_per_row) * GROUPS_PER_PROG

    offs_c = tl.arange(0, CG_C)  # within a group
    offs_g = tl.arange(0, GROUPS_PER_PROG)  # group offset
    # shape: (GROUPS_PER_PROG, CG_C)
    base_row = row * C
    n_off = grp_start * CG_C + offs_g[:, None] * CG_C + offs_c[None, :]

    x_ptrs = X_ptr + base_row + n_off
    x = tl.load(x_ptrs).to(tl.float32)

    n = CG.to(tl.float32)
    mean = tl.sum(x, axis=1) / n  # (GROUPS_PER_PROG,)
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / n
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W_ptr + n_off).to(tl.float32)
    b = tl.load(B_ptr + n_off).to(tl.float32)

    y = xc * rstd[:, None] * w + b
    tl.store(Y_ptr + base_row + n_off, y)


def group_norm(x, weight, bias, num_groups, eps):
    M, C = x.shape
    G = num_groups
    CG = C // G  # 32

    # pick GROUPS_PER_PROG so total tile size is reasonable
    GROUPS_PER_PROG = 8  # 8 groups * 32 = 256 cols per program
    if G % GROUPS_PER_PROG != 0:
        GROUPS_PER_PROG = 1
    progs_per_row = G // GROUPS_PER_PROG

    out = torch.empty_like(x, dtype=torch.float32)
    grid = (M * progs_per_row,)
    group_norm_kernel_packed[grid](
        x, weight, bias, out,
        M, C, G, CG, eps,
        GROUPS_PER_PROG=GROUPS_PER_PROG,
        CG_C=CG,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        self.num_groups = num_groups
        self.eps = 1e-5

    def forward(self, x):
        x = x.cuda().contiguous()
        W = self.gemm.weight
        b_lin = self.gemm.bias
        b_extra = self.bias
        y = gemm_bias_act(x, W, b_lin, b_extra)
        out = group_norm(y, self.groupnorm.weight, self.groupnorm.bias,
                         self.num_groups, self.eps)
        return out