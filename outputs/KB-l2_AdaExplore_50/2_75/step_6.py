import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
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


@triton.jit
def gn_min_kernel(
    X_ptr, gamma_ptr, beta_ptr, MIN_ptr,
    M, N, num_groups,
    eps: tl.constexpr,
    C_PER_G: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, C_PER_G)

    row_min = float('inf')
    for g in range(0, NUM_GROUPS):
        base = row * N + g * C_PER_G
        x = tl.load(X_ptr + base + offs)
        sum_x = tl.sum(x, axis=0)
        mean = sum_x / C_PER_G
        diff = x - mean
        var = tl.sum(diff * diff, axis=0) / C_PER_G
        rstd = 1.0 / tl.sqrt(var + eps)

        gamma = tl.load(gamma_ptr + g * C_PER_G + offs)
        beta = tl.load(beta_ptr + g * C_PER_G + offs)
        y = diff * rstd * gamma + beta
        gmin = tl.min(y, axis=0)
        row_min = tl.minimum(row_min, gmin)

    tl.store(MIN_ptr + row, row_min)


@triton.jit
def add_bias_kernel(
    min_ptr, bias_ptr, out_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    m = tl.load(min_ptr + row)
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
    out = m + b
    tl.store(out_ptr + row * N + offs, out, mask=mask)


def triton_gemm(x, weight_t, bias):
    """weight_t: (K, N) contiguous"""
    M, K = x.shape
    K2, N = weight_t.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_kernel[grid](
        x, weight_t, out, bias,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
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
        # cache transposed weight
        self._wt_cache = None

    def _get_wt(self):
        w = self.gemm.weight
        if (self._wt_cache is None
                or self._wt_cache.device != w.device
                or self._wt_cache.dtype != w.dtype
                or self._wt_cache.shape[0] != w.shape[1]):
            self._wt_cache = w.t().contiguous()
        return self._wt_cache

    def forward(self, x):
        x = x.contiguous()
        wt = self._get_wt()
        y = triton_gemm(x, wt, self.gemm.bias)

        M, N = y.shape
        C_per_g = N // self.num_groups

        min_buf = torch.empty((M,), device=y.device, dtype=y.dtype)

        gn_min_kernel[(M,)](
            y, self.group_norm.weight, self.group_norm.bias, min_buf,
            M, N, self.num_groups,
            eps=float(self.group_norm.eps),
            C_PER_G=C_per_g,
            NUM_GROUPS=self.num_groups,
            num_warps=4,
            num_stages=2,
        )

        out = min_buf.view(1, 1, M, 1) + self.bias
        return out