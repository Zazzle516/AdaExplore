import torch
import torch.nn as nn
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
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

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=['C_PER_G'],
)
@triton.jit
def group_norm_hardtanh_kernel(
    X_ptr, gamma_ptr, beta_ptr, Out_ptr,
    M, C, G, C_PER_G,
    eps,
    hmin: tl.constexpr, hmax: tl.constexpr,
    stride_xm, stride_xc,
    BLOCK_CG: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    offs = tl.arange(0, BLOCK_CG)
    mask = offs < C_PER_G

    c_start = pid_g * C_PER_G
    x_ptrs = X_ptr + pid_m * stride_xm + (c_start + offs) * stride_xc
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_masked = tl.where(mask, x, 0.0)

    sum_x = tl.sum(x_masked, axis=0)
    mean = sum_x / C_PER_G
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / C_PER_G
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(gamma_ptr + c_start + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(beta_ptr + c_start + offs, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * g + b
    y = tl.minimum(tl.maximum(y, hmin), hmax)

    out_ptrs = Out_ptr + pid_m * stride_xm + (c_start + offs) * stride_xc
    tl.store(out_ptrs, y, mask=mask)


def triton_gemm(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    Bt = weight.t().contiguous()
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    gemm_kernel[grid](
        x, Bt, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        Bt.stride(0), Bt.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


def triton_gn_hardtanh(x, gamma, beta, num_groups, eps, hmin, hmax):
    M, C = x.shape
    C_per_G = C // num_groups
    BLOCK_CG = triton.next_power_of_2(C_per_G)
    out = torch.empty_like(x)
    grid = (M, num_groups)
    group_norm_hardtanh_kernel[grid](
        x, gamma, beta, out,
        M, C, num_groups, C_per_G,
        eps, hmin, hmax,
        x.stride(0), x.stride(1),
        BLOCK_CG=BLOCK_CG,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.num_groups = num_groups

    def forward(self, x):
        x = x.contiguous()
        y = triton_gemm(x, self.gemm.weight, self.gemm.bias)
        out = triton_gn_hardtanh(
            y,
            self.group_norm.weight, self.group_norm.bias,
            self.num_groups, self.group_norm.eps,
            self.hardtanh_min, self.hardtanh_max,
        )
        return out