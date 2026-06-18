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
    K2, N = weight_t.shape
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


@triton.jit
def fused_gn_swish_mul_swish_kernel(
    X_ptr, GAMMA_ptr, BETA_ptr, MW_ptr, OUT_ptr,
    N, C, G, CPG,
    eps,
    BLOCK_CPG: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    offs = tl.arange(0, BLOCK_CPG)
    mask = offs < CPG

    c_idx = pid_g * CPG + offs

    x_ptrs = X_ptr + pid_n * C + c_idx
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
    mean = sum_x / CPG
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / CPG
    rstd = tl.rsqrt(var + eps)

    gamma = tl.load(GAMMA_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(BETA_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    mw = tl.load(MW_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * gamma + beta
    s1 = tl.sigmoid(y)
    y1 = y * s1
    y2 = y1 * mw
    s2 = tl.sigmoid(y2)
    out = y2 * s2

    tl.store(OUT_ptr + pid_n * C + c_idx, out, mask=mask)


def fused_gn_swish_mul_swish(x, gamma, beta, mw, num_groups, eps=1e-5):
    N, C = x.shape
    G = num_groups
    CPG = C // G
    out = torch.empty_like(x)
    BLOCK_CPG = triton.next_power_of_2(CPG)
    grid = (N, G)
    fused_gn_swish_mul_swish_kernel[grid](
        x, gamma, beta, mw, out,
        N, C, G, CPG,
        eps,
        BLOCK_CPG=BLOCK_CPG,
        num_warps=1 if BLOCK_CPG <= 32 else (2 if BLOCK_CPG <= 128 else 4),
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

    def _get_weight_t(self):
        w = self.gemm.weight
        if (self._weight_t_cache is None
                or self._weight_t_cache.device != w.device
                or self._weight_t_cache.dtype != w.dtype
                or self._weight_t_cache.data_ptr() == 0):
            self._weight_t_cache = w.t().contiguous()
        return self._weight_t_cache

    def forward(self, x):
        x = x.contiguous().cuda()
        weight_t = self._get_weight_t()
        y = triton_linear(x, weight_t, self.gemm.bias)
        out = fused_gn_swish_mul_swish(
            y,
            self.group_norm.weight,
            self.group_norm.bias,
            self.multiply_weight,
            self.num_groups,
            self.eps,
        )
        return out