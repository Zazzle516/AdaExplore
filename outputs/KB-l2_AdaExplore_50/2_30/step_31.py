import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
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

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_linear(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    gemm_bias_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        out.stride(0), out.stride(1),
    )
    return out


@triton.jit
def group_norm_hardtanh_kernel(
    X_ptr, Y_ptr, gamma_ptr, beta_ptr,
    M, C, G, CPG,
    EPS: tl.constexpr,
    HMIN: tl.constexpr, HMAX: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    sample = pid // G
    group = pid % G

    base = sample * C + group * CPG
    ch_base = group * CPG

    # First pass: compute sum and sum of squares using tiles
    sum_x = tl.zeros((), dtype=tl.float32)
    sum_x2 = tl.zeros((), dtype=tl.float32)

    for i in tl.static_range(0, CPG, TILE):
        offs = i + tl.arange(0, TILE)
        mask = offs < CPG
        x = tl.load(X_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    mean = sum_x / CPG
    var = sum_x2 / CPG - mean * mean
    rstd = 1.0 / tl.sqrt(var + EPS)

    # Second pass: normalize, affine, hardtanh, store
    for i in tl.static_range(0, CPG, TILE):
        offs = i + tl.arange(0, TILE)
        mask = offs < CPG
        x = tl.load(X_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(gamma_ptr + ch_base + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(beta_ptr + ch_base + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * g + b
        y = tl.minimum(tl.maximum(y, HMIN), HMAX)
        tl.store(Y_ptr + base + offs, y, mask=mask)


def triton_groupnorm_hardtanh(x, gamma, beta, num_groups, eps, hmin, hmax):
    M, C = x.shape
    CPG = C // num_groups
    out = torch.empty_like(x)
    grid = (M * num_groups,)
    # Choose tile size
    if CPG >= 512:
        TILE = 512
        num_warps = 8
    elif CPG >= 256:
        TILE = 256
        num_warps = 4
    elif CPG >= 128:
        TILE = 128
        num_warps = 4
    else:
        TILE = triton.next_power_of_2(CPG)
        num_warps = 2
    group_norm_hardtanh_kernel[grid](
        x, out, gamma, beta,
        M, C, num_groups, CPG,
        EPS=eps, HMIN=hmin, HMAX=hmax,
        BLOCK=TILE, TILE=TILE,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.num_groups = num_groups
        self.hardtanh_min = float(hardtanh_min)
        self.hardtanh_max = float(hardtanh_max)
        self.eps = 1e-5

    def forward(self, x):
        x = x.cuda().contiguous()
        weight = self.gemm.weight.contiguous()
        bias = self.gemm.bias.contiguous()
        y = triton_linear(x, weight, bias)
        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()
        out = triton_groupnorm_hardtanh(
            y, gamma, beta, self.num_groups, self.eps,
            self.hardtanh_min, self.hardtanh_max
        )
        return out