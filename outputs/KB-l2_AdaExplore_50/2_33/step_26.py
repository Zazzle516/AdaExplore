import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
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
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)
    acc = (acc + bias[None, :]) * scale[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=mask)


@triton.jit
def bn_stats_apply_kernel(
    X_ptr, Y_ptr,
    weight_ptr, bias_ptr,
    running_mean_ptr, running_var_ptr,
    M, N,
    eps, momentum, inv_M, var_factor,
    BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)

    # Pass 1: compute sum and sum of squares
    sum_x = tl.zeros((BLOCK_M,), dtype=tl.float32)
    sum_x2 = tl.zeros((BLOCK_M,), dtype=tl.float32)

    base = X_ptr + pid_n
    for m_start in range(0, M, BLOCK_M):
        m_idx = m_start + offs_m
        mask = m_idx < M
        x = tl.load(base + m_idx * N, mask=mask, other=0.0)
        sum_x += x
        sum_x2 += x * x

    s = tl.sum(sum_x, axis=0)
    s2 = tl.sum(sum_x2, axis=0)

    mean = s * inv_M
    var = s2 * inv_M - mean * mean

    w = tl.load(weight_ptr + pid_n)
    b = tl.load(bias_ptr + pid_n)
    inv_std = 1.0 / tl.sqrt(var + eps)
    scale = w * inv_std
    shift = b - mean * scale

    # Update running stats
    rm = tl.load(running_mean_ptr + pid_n)
    rv = tl.load(running_var_ptr + pid_n)
    new_rm = (1.0 - momentum) * rm + momentum * mean
    new_rv = (1.0 - momentum) * rv + momentum * var * var_factor
    tl.store(running_mean_ptr + pid_n, new_rm)
    tl.store(running_var_ptr + pid_n, new_rv)

    # Pass 2: normalize
    base_y = Y_ptr + pid_n
    for m_start in range(0, M, BLOCK_M):
        m_idx = m_start + offs_m
        mask = m_idx < M
        x = tl.load(base + m_idx * N, mask=mask, other=0.0)
        y = x * scale + shift
        tl.store(base_y + m_idx * N, y, mask=mask)


@triton.jit
def bn_eval_apply_kernel(
    X_ptr, Y_ptr,
    weight_ptr, bias_ptr,
    running_mean_ptr, running_var_ptr,
    M, N, eps,
    BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)

    rm = tl.load(running_mean_ptr + pid_n)
    rv = tl.load(running_var_ptr + pid_n)
    w = tl.load(weight_ptr + pid_n)
    b = tl.load(bias_ptr + pid_n)

    inv_std = 1.0 / tl.sqrt(rv + eps)
    scale = w * inv_std
    shift = b - rm * scale

    base_x = X_ptr + pid_n
    base_y = Y_ptr + pid_n
    for m_start in range(0, M, BLOCK_M):
        m_idx = m_start + offs_m
        mask = m_idx < M
        x = tl.load(base_x + m_idx * N, mask=mask, other=0.0)
        y = x * scale + shift
        tl.store(base_y + m_idx * N, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.momentum = momentum

        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        x = x.contiguous()
        M, K = x.shape
        N = self.out_features

        W = self.gemm.weight  # (N, K)
        bias = self.gemm.bias  # (N,)
        scale = self.scale  # (N,)

        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
        gemm_scale_kernel[grid](
            x, W, bias, scale, out,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(1), W.stride(0),
            out.stride(0), out.stride(1),
        )

        y = torch.empty_like(out)

        BLOCK_M = 256
        if self.training:
            inv_M = 1.0 / float(M)
            var_factor = float(M) / float(max(M - 1, 1))
            bn_stats_apply_kernel[(N,)](
                out, y,
                self.bn.weight, self.bn.bias,
                self.bn.running_mean, self.bn.running_var,
                M, N,
                self.eps, self.momentum, inv_M, var_factor,
                BLOCK_M=BLOCK_M,
                num_warps=4,
            )
        else:
            bn_eval_apply_kernel[(N,)](
                out, y,
                self.bn.weight, self.bn.bias,
                self.bn.running_mean, self.bn.running_var,
                M, N, self.eps,
                BLOCK_M=BLOCK_M,
                num_warps=4,
            )
        return y