import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_bias_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
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

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a_mask = mask_m[:, None] & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & mask_n[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_linear(x, weight_t, bias):
    M, K = x.shape
    _, N = weight_t.shape
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_bias_kernel[grid](
        x, weight_t, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


# Fused GroupNorm + Swish + Mul + Swish
GN_CONFIGS = [
    triton.Config({'ROWS_PER_PROG': 4}, num_warps=2, num_stages=2),
    triton.Config({'ROWS_PER_PROG': 8}, num_warps=4, num_stages=2),
    triton.Config({'ROWS_PER_PROG': 16}, num_warps=4, num_stages=2),
    triton.Config({'ROWS_PER_PROG': 32}, num_warps=8, num_stages=2),
    triton.Config({'ROWS_PER_PROG': 16}, num_warps=8, num_stages=2),
    triton.Config({'ROWS_PER_PROG': 8}, num_warps=2, num_stages=2),
]


@triton.autotune(configs=GN_CONFIGS, key=['N', 'C', 'CPG'])
@triton.jit
def fused_gn_swish_mul_swish_kernel(
    X_ptr, GAMMA_ptr, BETA_ptr, MW_ptr, OUT_ptr,
    N, C,
    eps,
    CPG: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid_row_block = tl.program_id(0)
    pid_g = tl.program_id(1)

    row_start = pid_row_block * ROWS_PER_PROG
    rows = row_start + tl.arange(0, ROWS_PER_PROG)
    row_mask = rows < N

    cols = tl.arange(0, CPG)
    c_idx = pid_g * CPG + cols

    offs = rows[:, None] * C + c_idx[None, :]
    mask = row_mask[:, None]

    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # E[x^2] - E[x]^2 formulation
    sum_x = tl.sum(xf, axis=1)
    sum_xx = tl.sum(xf * xf, axis=1)
    mean = sum_x / CPG
    var = sum_xx / CPG - mean * mean
    rstd = tl.rsqrt(var + eps)

    gamma = tl.load(GAMMA_ptr + c_idx)
    beta = tl.load(BETA_ptr + c_idx)
    mw = tl.load(MW_ptr + c_idx)

    y = (xf - mean[:, None]) * rstd[:, None] * gamma[None, :] + beta[None, :]
    s1 = tl.sigmoid(y)
    y1 = y * s1
    y2 = y1 * mw[None, :]
    s2 = tl.sigmoid(y2)
    out = y2 * s2

    tl.store(OUT_ptr + offs, out, mask=mask)


def fused_gn_swish_mul_swish(x, gamma, beta, mw, num_groups, eps=1e-5):
    N, C = x.shape
    G = num_groups
    CPG = C // G
    out = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(N, meta['ROWS_PER_PROG']), G)
    fused_gn_swish_mul_swish_kernel[grid](
        x, gamma, beta, mw, out,
        N, C,
        eps,
        CPG=CPG,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        self.num_groups = num_groups
        self.eps = 1e-5
        self._weight_t_cache = None
        self._weight_data_ptr = None

    def _get_weight_t(self, device):
        w = self.gemm.weight
        cur_ptr = w.data_ptr()
        if (self._weight_t_cache is None or
                self._weight_data_ptr != cur_ptr or
                self._weight_t_cache.device != device):
            self._weight_t_cache = w.detach().t().contiguous().to(device)
            self._weight_data_ptr = cur_ptr
        return self._weight_t_cache

    def forward(self, x):
        x = x.contiguous().cuda()
        weight_t = self._get_weight_t(x.device)
        bias = self.gemm.bias.contiguous()
        y = triton_linear(x, weight_t, bias)
        out = fused_gn_swish_mul_swish(
            y,
            self.group_norm.weight,
            self.group_norm.bias,
            self.multiply_weight,
            self.num_groups,
            self.eps,
        )
        return out