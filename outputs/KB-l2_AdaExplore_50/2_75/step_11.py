import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
</old_str_1>
<new_str_1>
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr, Out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
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
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c = tl.load(C_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + c[None, :]
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_linear(x, weight_t, bias):
    # weight_t is (K, N) — pre-transposed weight
    M, K = x.shape
    K2, N = weight_t.shape
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_kernel[grid](
        x, weight_t, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


@triton.jit
def gn_groupmin_kernel(
    X_ptr, Gamma_ptr, Beta_ptr, GMin_ptr,
    M, C, G, CPG,
    eps,
    GROUPS_PER_PROG: tl.constexpr,
    BLOCK_CPG: tl.constexpr,
):
    row = tl.program_id(0)
    gblk = tl.program_id(1)
    g_start = gblk * GROUPS_PER_PROG

    offs = tl.arange(0, BLOCK_CPG)
    mask_cpg = offs < CPG

    for gi in tl.static_range(GROUPS_PER_PROG):
        g = g_start + gi
        x_ptrs = X_ptr + row * C + g * CPG + offs
        x = tl.load(x_ptrs, mask=mask_cpg, other=0.0)
        sum_x = tl.sum(x, axis=0)
        mean = sum_x / CPG
        xc = tl.where(mask_cpg, x - mean, 0.0)
        var = tl.sum(xc * xc, axis=0) / CPG
        rstd = 1.0 / tl.sqrt(var + eps)
        gamma = tl.load(Gamma_ptr + g * CPG + offs, mask=mask_cpg, other=0.0)
        beta = tl.load(Beta_ptr + g * CPG + offs, mask=mask_cpg, other=0.0)
        y = xc * rstd * gamma + beta
        y_masked = tl.where(mask_cpg, y, float('inf'))
        group_min = tl.min(y_masked, axis=0)
        tl.store(GMin_ptr + row * G + g, group_min)


@triton.jit
def reduce_min_add_bias_kernel(
    GMin_ptr, Bias_ptr, Out_ptr,
    M, C, G,
    BLOCK_G: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    col_block = tl.program_id(1)

    # Reduce min across G groups for this row
    offs_g = tl.arange(0, BLOCK_G)
    mask_g = offs_g < G
    g_vals = tl.load(GMin_ptr + row * G + offs_g, mask=mask_g, other=float('inf'))
    row_min = tl.min(g_vals, axis=0)

    offs_c = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    b = tl.load(Bias_ptr + offs_c, mask=mask_c, other=0.0)
    tl.store(Out_ptr + row * C + offs_c, row_min + b, mask=mask_c)


def triton_gn_min_bias(x, gamma, beta, bias_flat, num_groups, eps):
    M, C = x.shape
    CPG = C // num_groups
    G = num_groups
    BLOCK_CPG = triton.next_power_of_2(CPG)

    # Group multiple groups per program to amortize launch overhead
    GROUPS_PER_PROG = 8
    while G % GROUPS_PER_PROG != 0 and GROUPS_PER_PROG > 1:
        GROUPS_PER_PROG //= 2

    gmin = torch.empty((M, G), device=x.device, dtype=x.dtype)
    grid1 = (M, G // GROUPS_PER_PROG)
    gn_groupmin_kernel[grid1](
        x, gamma, beta, gmin,
        M, C, G, CPG,
        eps,
        GROUPS_PER_PROG=GROUPS_PER_PROG,
        BLOCK_CPG=BLOCK_CPG,
        num_warps=2,
        num_stages=2,
    )

    out = torch.empty((M, C), device=x.device, dtype=x.dtype)
    BLOCK_C = 512
    BLOCK_G = triton.next_power_of_2(G)
    grid2 = (M, triton.cdiv(C, BLOCK_C))
    reduce_min_add_bias_kernel[grid2](
        gmin, bias_flat, out,
        M, C, G,
        BLOCK_G=BLOCK_G,
        BLOCK_C=BLOCK_C,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.num_groups = num_groups
        self.out_features = out_features
        self.in_features = in_features

        # Pre-transpose weight to (K, N) contiguous for better B access pattern
        with torch.no_grad():
            wt = self.gemm.weight.t().contiguous()
        self.register_buffer('weight_t', wt, persistent=False)

    def forward(self, x):
        x = x.contiguous()
        # Refresh weight_t in case weight was modified (best effort)
        if self.weight_t.shape != (self.in_features, self.out_features) or \
           self.weight_t.device != self.gemm.weight.device:
            self.weight_t = self.gemm.weight.t().contiguous()

        b = self.gemm.bias.contiguous()
        y = triton_linear(x, self.weight_t, b)

        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()
        eps = self.group_norm.eps

        bias_flat = self.bias.view(-1).contiguous()
        out = triton_gn_min_bias(y, gamma, beta, bias_flat, self.num_groups, eps)
        # Output shape: (M, C, 1, 1)
        return out.view(x.shape[0], self.out_features, 1, 1)