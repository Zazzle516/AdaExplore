import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


MATMUL_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=MATMUL_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def matmul_swish_bias_kernel(
    A_ptr, B_ptr, C_ptr, bias_lin_ptr, bias_extra_ptr,
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
        a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # add linear bias
    bias_lin = tl.load(bias_lin_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias_lin[None, :]

    # swish: x * sigmoid(x) -- compute in fp32 explicitly
    sig = 1.0 / (1.0 + tl.exp(-acc))
    acc = acc * sig

    # add extra bias
    bias_extra = tl.load(bias_extra_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias_extra[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=mask)


@triton.jit
def group_norm_kernel(
    X_ptr, Y_ptr, weight_ptr, bias_ptr,
    M, C, num_groups, group_size, eps,
    BLOCK: tl.constexpr,
):
    # one program per (sample, group)
    pid = tl.program_id(0)
    sample = pid // num_groups
    group = pid % num_groups

    offs = tl.arange(0, BLOCK)
    mask = offs < group_size

    base = sample * C + group * group_size
    x_ptrs = X_ptr + base + offs

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_x = tl.sum(x, axis=0)
    mean = sum_x / group_size

    xc = tl.where(mask, x - mean, 0.0)
    sum_xx = tl.sum(xc * xc, axis=0)
    var = sum_xx / group_size
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(weight_ptr + group * group_size + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + group * group_size + offs, mask=mask, other=0.0)

    y = xc * rstd * w + b

    tl.store(Y_ptr + base + offs, y, mask=mask)


def matmul_swish_bias(x, weight, bias_lin, bias_extra):
    M, K = x.shape
    N = weight.shape[0]
    # weight is (out_features, in_features); we want B = weight.T -> (K, N)
    # Use weight directly with adjusted strides
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # B_ptr: pretend weight is (K, N) with stride_bk = 1, stride_bn = K (since weight is (N, K) contig)
    stride_am, stride_ak = x.stride()
    # weight shape (N, K), strides (K, 1). For B (K, N): B[k, n] = weight[n, k]
    stride_bk = 1
    stride_bn = weight.stride(0)  # = K
    stride_cm, stride_cn = out.stride()

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    matmul_swish_bias_kernel[grid](
        x, weight, out, bias_lin, bias_extra,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        GROUP_M=8,
    )
    return out


def group_norm_fwd(x, weight, bias, num_groups, eps):
    M, C = x.shape
    group_size = C // num_groups
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(group_size)
    grid = (M * num_groups,)
    nw = 2 if BLOCK <= 64 else (4 if BLOCK <= 256 else 8)
    group_norm_kernel[grid](
        x, out, weight, bias,
        M, C, num_groups, group_size, eps,
        BLOCK=BLOCK,
        num_warps=nw,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.num_groups = num_groups
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous()
        w = self.matmul.weight.contiguous()
        b_lin = self.matmul.bias.contiguous()
        b_extra = self.bias.contiguous()
        y = matmul_swish_bias(x, w, b_lin, b_extra)
        y = group_norm_fwd(y, self.group_norm.weight, self.group_norm.bias,
                           self.num_groups, self.group_norm.eps)
        return y