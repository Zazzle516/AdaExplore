import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_bias_scale_kernel(
    A, B, Bias, Scale, C,
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

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # bias + scale
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias = tl.load(Bias + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
    scale = tl.load(Scale + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
    acc = (acc + bias[None, :]) * scale[None, :]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def gemm_bias_scale(x, w, bias, scale):
    # x: (M, K), w: (N, K) -> out: (M, N) = x @ w.T + bias, then * scale
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_bias_scale_kernel[grid](
        x, w, bias, scale, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(1), w.stride(0),  # B = w.T, so stride_bk = w.stride(1), stride_bn = w.stride(0)
        out.stride(0), out.stride(1),
    )
    return out


# BatchNorm: compute mean/var along dim 0, then normalize.
@triton.jit
def bn_stats_kernel(
    X, Mean, Var,
    M, N,
    stride_xm, stride_xn,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)  # column index
    if pid >= N:
        return
    offs_m = tl.arange(0, BLOCK_M)
    sum_x = tl.zeros((BLOCK_M,), dtype=tl.float32)
    sum_xx = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        idx = m_start + offs_m
        mask = idx < M
        x = tl.load(X + idx * stride_xm + pid * stride_xn, mask=mask, other=0.0)
        sum_x += tl.where(mask, x, 0.0)
        sum_xx += tl.where(mask, x * x, 0.0)
    s = tl.sum(sum_x, axis=0)
    ss = tl.sum(sum_xx, axis=0)
    mean = s / M
    var = ss / M - mean * mean
    tl.store(Mean + pid, mean)
    tl.store(Var + pid, var)


@triton.jit
def bn_apply_kernel(
    X, Out, Mean, Var, Gamma, Beta,
    M, N, eps,
    stride_xm, stride_xn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mean = tl.load(Mean + offs_n, mask=mask_n, other=0.0)
    var = tl.load(Var + offs_n, mask=mask_n, other=0.0)
    gamma = tl.load(Gamma + offs_n, mask=mask_n, other=0.0)
    beta = tl.load(Beta + offs_n, mask=mask_n, other=0.0)
    inv = 1.0 / tl.sqrt(var + eps)
    scale = gamma * inv
    shift = beta - mean * scale

    x_ptrs = X + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    o_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask = mask_m[:, None] & mask_n[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    y = x * scale[None, :] + shift[None, :]
    tl.store(o_ptrs, y, mask=mask)


def batchnorm_train(x, gamma, beta, running_mean, running_var, eps, momentum):
    M, N = x.shape
    mean = torch.empty(N, device=x.device, dtype=torch.float32)
    var = torch.empty(N, device=x.device, dtype=torch.float32)
    BLOCK_M = 1024
    bn_stats_kernel[(N,)](
        x, mean, var, M, N,
        x.stride(0), x.stride(1),
        BLOCK_M=BLOCK_M, num_warps=8,
    )
    out = torch.empty_like(x)
    BM = 64
    BN = 128
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    bn_apply_kernel[grid](
        x, out, mean, var, gamma, beta,
        M, N, eps,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, num_warps=4,
    )
    # update running stats (unbiased var for running)
    with torch.no_grad():
        running_mean.mul_(1 - momentum).add_(mean, alpha=momentum)
        unbiased_var = var * (M / max(M - 1, 1))
        running_var.mul_(1 - momentum).add_(unbiased_var, alpha=momentum)
    return out


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
        w = self.gemm.weight.contiguous()
        b = self.gemm.bias.contiguous()
        s = self.scale.contiguous().view(-1)
        y = gemm_bias_scale(x, w, b, s)
        if self.training:
            out = batchnorm_train(
                y, self.bn.weight, self.bn.bias,
                self.bn.running_mean, self.bn.running_var,
                self.eps, self.momentum,
            )
        else:
            mean = self.bn.running_mean
            var = self.bn.running_var
            out = torch.empty_like(y)
            M, N = y.shape
            BM = 64
            BN = 128
            grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
            bn_apply_kernel[grid](
                y, out, mean, var, self.bn.weight, self.bn.bias,
                M, N, self.eps,
                y.stride(0), y.stride(1),
                out.stride(0), out.stride(1),
                BLOCK_M=BM, BLOCK_N=BN, num_warps=4,
            )
        return out