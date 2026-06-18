import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_scale_kernel(
    A, B, Bias, Scale, Out,
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
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_k[None, :] < k_remaining) & mask_m[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(Bias + offs_n, mask=mask_n, other=0.0)
    scale = tl.load(Scale + offs_n, mask=mask_n, other=0.0)
    acc = (acc + bias[None, :]) * scale[None, :]

    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def gemm_scale(x, w, b, scale):
    M, K = x.shape
    N, _ = w.shape
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_scale_kernel[grid](
        x, w, b, scale, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(1), w.stride(0),  # w is (N,K), we want B[k,n] => stride_bk = w.stride(1), stride_bn = w.stride(0)
        out.stride(0), out.stride(1),
    )
    return out


@triton.jit
def bn_stats_kernel(
    X, Mean, Var,
    M, N,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)  # column index
    offs_m = tl.arange(0, BLOCK_M)
    sum_x = tl.zeros((BLOCK_M,), dtype=tl.float32)
    sum_x2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        offs = m_start + offs_m
        mask = offs < M
        x = tl.load(X + offs * N + pid, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.where(mask, x, 0.0)
        sum_x2 += tl.where(mask, x * x, 0.0)
    s = tl.sum(sum_x, axis=0)
    s2 = tl.sum(sum_x2, axis=0)
    mean = s / M
    var = s2 / M - mean * mean
    tl.store(Mean + pid, mean)
    tl.store(Var + pid, var)


@triton.jit
def bn_apply_kernel(
    X, Out, Mean, Var, Weight, Bias,
    M, N, eps,
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
    w = tl.load(Weight + offs_n, mask=mask_n, other=0.0)
    b = tl.load(Bias + offs_n, mask=mask_n, other=0.0)
    inv = 1.0 / tl.sqrt(var + eps)
    scale = w * inv
    shift = b - mean * scale

    ptrs = offs_m[:, None] * N + offs_n[None, :]
    x = tl.load(X + ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    y = x * scale[None, :] + shift[None, :]
    tl.store(Out + ptrs, y, mask=mask_m[:, None] & mask_n[None, :])


def batchnorm_train(x, weight, bias, running_mean, running_var, eps, momentum):
    M, N = x.shape
    mean = torch.empty(N, device=x.device, dtype=torch.float32)
    var = torch.empty(N, device=x.device, dtype=torch.float32)
    BLOCK_M_STATS = 1024
    bn_stats_kernel[(N,)](x, mean, var, M, N, BLOCK_M=BLOCK_M_STATS)

    # update running stats
    with torch.no_grad():
        running_mean.mul_(1 - momentum).add_(mean, alpha=momentum)
        # unbiased var for running
        unbiased_var = var * (M / max(M - 1, 1))
        running_var.mul_(1 - momentum).add_(unbiased_var, alpha=momentum)

    out = torch.empty_like(x)
    BLOCK_M = 64
    BLOCK_N = 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    bn_apply_kernel[grid](x, out, mean, var, weight, bias, M, N, eps,
                          BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
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
        x = x.contiguous()
        scale = self.scale.view(-1).contiguous()
        w = self.gemm.weight.contiguous()
        b = self.gemm.bias.contiguous()
        y = gemm_scale(x, w, b, scale)

        if self.training:
            return batchnorm_train(y, self.bn.weight, self.bn.bias,
                                   self.bn.running_mean, self.bn.running_var,
                                   self.eps, self.momentum)
        else:
            # Eval: fold into elementwise
            inv = 1.0 / torch.sqrt(self.bn.running_var + self.eps)
            s = self.bn.weight * inv
            sh = self.bn.bias - self.bn.running_mean * s
            return y * s + sh