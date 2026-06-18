import torch
import torch.nn as nn
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_scale_kernel(
    A_ptr, B_ptr, bias_ptr, scale_ptr, C_ptr,
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
        a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)
    acc = (acc + bias[None, :]) * scale[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(mask_m[:, None]) & (mask_n[None, :]))


@triton.jit
def bn_stats_kernel(
    X_ptr, mean_ptr, var_ptr,
    M, N,
    stride_m, stride_n,
    BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    sum_x = tl.zeros((), dtype=tl.float32)
    sum_x2 = tl.zeros((), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        cur = m_start + offs_m
        mask = cur < M
        x = tl.load(X_ptr + cur * stride_m + pid_n * stride_n, mask=mask, other=0.0)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)
    mean = sum_x / M
    var = sum_x2 / M - mean * mean
    tl.store(mean_ptr + pid_n, mean)
    tl.store(var_ptr + pid_n, var)


@triton.jit
def bn_apply_kernel(
    X_ptr, Y_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    M, N, eps,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    mean = tl.load(mean_ptr + offs_n, mask=mask_n, other=0.0)
    var = tl.load(var_ptr + offs_n, mask=mask_n, other=0.0)
    gamma = tl.load(gamma_ptr + offs_n, mask=mask_n, other=0.0)
    beta = tl.load(beta_ptr + offs_n, mask=mask_n, other=0.0)
    inv = 1.0 / tl.sqrt(var + eps)
    scale = gamma * inv
    shift = beta - mean * scale

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    y = x * scale[None, :] + shift[None, :]
    tl.store(y_ptrs, y, mask=mask_m[:, None] & mask_n[None, :])


def fused_gemm_scale(x, weight, bias, scale):
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # B is weight.T => use weight with stride trick
    # A: x (M, K), B: weight.T (K, N) so B[k,n] = weight[n,k]
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    gemm_scale_kernel[grid](
        x, weight, bias, scale, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        out.stride(0), out.stride(1),
    )
    return out


def fused_bn(x, gamma, beta, running_mean, running_var, eps, momentum, training):
    M, N = x.shape
    y = torch.empty_like(x)
    if training:
        mean = torch.empty(N, device=x.device, dtype=torch.float32)
        var = torch.empty(N, device=x.device, dtype=torch.float32)
        bn_stats_kernel[(N,)](
            x, mean, var, M, N,
            x.stride(0), x.stride(1),
            BLOCK_M=1024,
        )
        with torch.no_grad():
            running_mean.mul_(1 - momentum).add_(mean, alpha=momentum)
            # unbiased var for running stats
            unbiased = var * (M / max(M - 1, 1))
            running_var.mul_(1 - momentum).add_(unbiased, alpha=momentum)
        use_mean = mean
        use_var = var
    else:
        use_mean = running_mean
        use_var = running_var

    BLOCK_M = 64
    BLOCK_N = 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    bn_apply_kernel[grid](
        x, y, use_mean, use_var, gamma, beta,
        M, N, eps,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.gemm.weight.contiguous()
        bias = self.gemm.bias.contiguous()
        scale = self.scale.contiguous().view(-1)
        out = fused_gemm_scale(x, weight, bias, scale)
        y = fused_bn(
            out,
            self.bn.weight, self.bn.bias,
            self.bn.running_mean, self.bn.running_var,
            self.eps, self.momentum, self.training,
        )
        return y