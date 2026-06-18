import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_bias_kernel(
    A_ptr, B_ptr, C_ptr, bias_ptr,
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

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        mask_k = offs_k < (K - k)
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :].to(tl.float32)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def gn_lrelu_double_kernel(
    X_ptr, Y_ptr, gamma_ptr, beta_ptr,
    M, C, G, CPG,
    eps,
    negative_slope,
    BLOCK_C: tl.constexpr,
):
    # one program per (sample, group)
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    # channel offsets within group
    offs = tl.arange(0, BLOCK_C)
    mask = offs < CPG

    base = pid_m * C + pid_g * CPG
    x = tl.load(X_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    # mean and var
    cnt = tl.sum(mask.to(tl.float32), axis=0)
    s = tl.sum(x, axis=0)
    mean = s / cnt
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / cnt
    inv = 1.0 / tl.sqrt(var + eps)

    g = tl.load(gamma_ptr + pid_g * CPG + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(beta_ptr + pid_g * CPG + offs, mask=mask, other=0.0).to(tl.float32)

    y = xc * inv * g + b
    # leaky relu
    y = tl.where(y >= 0, y, y * negative_slope)
    # x + x  -> 2*y
    y = y + y

    tl.store(Y_ptr + base + offs, y, mask=mask)


def triton_linear(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # B is weight.T, shape (K, N); use weight strides
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    gemm_bias_kernel[grid](
        x, weight, out, bias,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # B = W^T, so stride_bk = W.stride(1), stride_bn = W.stride(0)
        out.stride(0), out.stride(1),
    )
    return out


def triton_gn_lrelu_double(x, gamma, beta, num_groups, eps, negative_slope):
    M, C = x.shape
    G = num_groups
    CPG = C // G
    out = torch.empty_like(x)
    BLOCK_C = triton.next_power_of_2(CPG)
    if BLOCK_C < 16:
        BLOCK_C = 16
    grid = (M, G)
    gn_lrelu_double_kernel[grid](
        x, out, gamma, beta,
        M, C, G, CPG,
        eps, negative_slope,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.fc.weight.contiguous()
        b = self.fc.bias.contiguous()
        y = triton_linear(x, w, b)
        gamma = self.gn.weight.contiguous()
        beta = self.gn.bias.contiguous()
        out = triton_gn_lrelu_double(y, gamma, beta, self.num_groups, self.eps, self.negative_slope)
        return out