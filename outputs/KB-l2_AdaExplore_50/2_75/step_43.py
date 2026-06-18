import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_bias_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
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

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

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

    bias = tl.load(Bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def gn_min_kernel(
    X_ptr, Gamma_ptr, Beta_ptr, Out_ptr,
    M, N,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    eps,
):
    row = tl.program_id(0)

    # 2D tile: [NUM_GROUPS, GROUP_SIZE]
    offs_g = tl.arange(0, NUM_GROUPS)[:, None]
    offs_s = tl.arange(0, GROUP_SIZE)[None, :]
    offs2d = offs_g * GROUP_SIZE + offs_s  # [NUM_GROUPS, GROUP_SIZE]

    x = tl.load(X_ptr + row * N + offs2d)
    gamma = tl.load(Gamma_ptr + offs2d)
    beta = tl.load(Beta_ptr + offs2d)

    # Per-group statistics: reduce along axis=1 (within group)
    s = tl.sum(x, axis=1)               # [NUM_GROUPS]
    sq = tl.sum(x * x, axis=1)          # [NUM_GROUPS]
    mean = s / GROUP_SIZE               # [NUM_GROUPS]
    var = sq / GROUP_SIZE - mean * mean # [NUM_GROUPS]
    inv = 1.0 / tl.sqrt(var + eps)      # [NUM_GROUPS]

    # Normalize
    norm = (x - mean[:, None]) * inv[:, None] * gamma + beta  # [NUM_GROUPS, GROUP_SIZE]

    # Min across both axes
    row_min = tl.min(tl.min(norm, axis=1), axis=0)

    tl.store(Out_ptr + row, row_min)


@triton.jit
def gn_min_bias_kernel(
    X_ptr, Gamma_ptr, Beta_ptr, Bias_ptr, Out_ptr,
    M, N,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    eps,
    BLOCK_BIAS: tl.constexpr,
):
    row = tl.program_id(0)

    # 2D tile: [NUM_GROUPS, GROUP_SIZE]
    offs_g = tl.arange(0, NUM_GROUPS)[:, None]
    offs_s = tl.arange(0, GROUP_SIZE)[None, :]
    offs2d = offs_g * GROUP_SIZE + offs_s

    x = tl.load(X_ptr + row * N + offs2d)
    gamma = tl.load(Gamma_ptr + offs2d)
    beta = tl.load(Beta_ptr + offs2d)

    s = tl.sum(x, axis=1)
    sq = tl.sum(x * x, axis=1)
    mean = s / GROUP_SIZE
    var = sq / GROUP_SIZE - mean * mean
    inv = 1.0 / tl.sqrt(var + eps)

    norm = (x - mean[:, None]) * inv[:, None] * gamma + beta
    row_min = tl.min(tl.min(norm, axis=1), axis=0)

    # write to Out [1, N, M, 1] -> flat index n*M + row, for n in [0,N)
    for nb in range(0, tl.cdiv(N, BLOCK_BIAS)):
        offs_n = nb * BLOCK_BIAS + tl.arange(0, BLOCK_BIAS)
        mask_n = offs_n < N
        bias = tl.load(Bias_ptr + offs_n, mask=mask_n, other=0.0)
        tl.store(Out_ptr + offs_n * M + row, row_min + bias, mask=mask_n)


@triton.jit
def add_bias_kernel(
    MinVal_ptr, Bias_ptr, Out_ptr,
    M, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = M * N
    mask = offs < total
    # output shape [1, N, M, 1]; index = n * M + m
    n_idx = offs // M
    m_idx = offs % M
    minv = tl.load(MinVal_ptr + m_idx, mask=mask, other=0.0)
    bias = tl.load(Bias_ptr + n_idx, mask=mask, other=0.0)
    tl.store(Out_ptr + offs, minv + bias, mask=mask)


def triton_gemm_bias(x, w_t, bias):
    M, K = x.shape
    K2, N = w_t.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_bias_kernel[grid](
        x, w_t, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.group_size = out_features // num_groups
        self.eps = 1e-5

        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._wt_cache = None

    def _get_wt(self):
        w = self.gemm.weight
        if (self._wt_cache is None
                or self._wt_cache.device != w.device
                or self._wt_cache.dtype != w.dtype
                or self._wt_cache.data_ptr() == 0):
            self._wt_cache = w.t().contiguous()
        return self._wt_cache

    def forward(self, x):
        M = x.shape[0]
        N = self.out_features

        x = x.contiguous()
        w_t = self._get_wt()
        gemm_out = triton_gemm_bias(x, w_t, self.gemm.bias)

        bias_flat = self.bias.view(-1)
        out = torch.empty((1, N, M, 1), device=x.device, dtype=x.dtype)

        BLOCK_BIAS = 1024
        gn_min_bias_kernel[(M,)](
            gemm_out, self.group_norm.weight, self.group_norm.bias, bias_flat, out,
            M, N,
            self.num_groups, self.group_size,
            self.eps,
            BLOCK_BIAS=BLOCK_BIAS,
            num_warps=4,
            num_stages=3,
        )
        return out