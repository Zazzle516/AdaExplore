import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_affine_kernel(
    a_ptr, b_ptr, bias_ptr, alpha_ptr, beta_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
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

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    mask_n = offs_n < N
    mask_m = offs_m < M

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    alpha = tl.load(alpha_ptr + offs_n, mask=mask_n, other=0.0)
    beta = tl.load(beta_ptr + offs_n, mask=mask_n, other=0.0)

    # (acc + bias) * alpha + beta  =  acc*alpha + (bias*alpha + beta)
    acc = acc + bias[None, :]
    acc = acc * alpha[None, :] + beta[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def col_stats_kernel(
    x_ptr, mean_ptr, var_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    col_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = col_offs < N
    s = tl.zeros((BLOCK_N,), dtype=tl.float32)
    sq = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for m in range(0, M):
        x = tl.load(x_ptr + m * N + col_offs, mask=mask_n, other=0.0)
        s += x
        sq += x * x
    mean = s / M
    var = sq / M - mean * mean
    tl.store(mean_ptr + col_offs, mean, mask=mask_n)
    tl.store(var_ptr + col_offs, var, mask=mask_n)


@triton.jit
def softmax_kernel_single(
    x_ptr, out_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * N
    out_row_ptr = out_ptr + row * N

    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_row_ptr + offs, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(out_row_ptr + offs, out, mask=mask)


@triton.jit
def gemm_bias_kernel(
    a_ptr, b_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
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

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    mask_m = offs_m < M
    mask_n = offs_n < N
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def affine_softmax_kernel_single(
    x_ptr, a_ptr, b_ptr, out_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * N
    out_row_ptr = out_ptr + row * N

    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_row_ptr + offs, mask=mask, other=0.0)
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    v = a * x + b
    v = tl.where(mask, v, -float('inf'))
    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(out_row_ptr + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum

        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))

    def _build_ab(self, mean, var, weight_bn, bias_bn, scale):
        inv = torch.rsqrt(var + self.bn_eps)
        a = scale * weight_bn * inv
        b = scale * (bias_bn - weight_bn * mean * inv)
        return a.contiguous(), b.contiguous()

    def forward(self, x):
        x = x.contiguous().cuda()
        M, K = x.shape
        N = self.out_features

        weight = self.gemm.weight
        bias = self.gemm.bias

        if not self.training:
            # Eval path: fold affine into GEMM epilogue, single softmax kernel reads affine-transformed output.
            a, b = self._build_ab(self.bn.running_mean, self.bn.running_var,
                                  self.bn.weight, self.bn.bias, self.scale)
            lin_out = torch.empty((M, N), device=x.device, dtype=torch.float32)
            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
            gemm_bias_affine_kernel[grid](
                x, weight, bias, a, b, lin_out,
                M, N, K,
                x.stride(0), x.stride(1),
                weight.stride(1), weight.stride(0),
            )

            out = torch.empty_like(lin_out)
            BLOCK_N = 1
            while BLOCK_N < N:
                BLOCK_N *= 2
            softmax_kernel_single[(M,)](
                lin_out, out,
                M, N,
                BLOCK_N=BLOCK_N,
                num_warps=8,
                num_stages=2,
            )
            return out

        # Training path
        lin_out = torch.empty((M, N), device=x.device, dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
        gemm_bias_kernel[grid](
            x, weight, bias, lin_out,
            M, N, K,
            x.stride(0), x.stride(1),
            weight.stride(1), weight.stride(0),
        )

        mean = torch.empty(N, device=x.device, dtype=torch.float32)
        var = torch.empty(N, device=x.device, dtype=torch.float32)
        BLOCK_N_STAT = 256
        col_stats_kernel[(triton.cdiv(N, BLOCK_N_STAT),)](
            lin_out, mean, var,
            M, N,
            BLOCK_N=BLOCK_N_STAT,
            num_warps=4,
        )
        with torch.no_grad():
            unbiased_var = var * (M / (M - 1)) if M > 1 else var
            self.bn.running_mean.mul_(1 - self.bn_momentum).add_(mean, alpha=self.bn_momentum)
            self.bn.running_var.mul_(1 - self.bn_momentum).add_(unbiased_var, alpha=self.bn_momentum)
            self.bn.num_batches_tracked.add_(1)

        a, b = self._build_ab(mean, var, self.bn.weight, self.bn.bias, self.scale)

        out = torch.empty_like(lin_out)
        BLOCK_N = 1
        while BLOCK_N < N:
            BLOCK_N *= 2
        affine_softmax_kernel_single[(M,)](
            lin_out, a, b, out,
            M, N,
            BLOCK_N=BLOCK_N,
            num_warps=8,
            num_stages=2,
        )
        return out