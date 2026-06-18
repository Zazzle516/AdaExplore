import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
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
    acc += bias[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def col_mean_var_kernel(
    x_ptr, mean_ptr, var_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
):
    # one program per column
    col = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    sum_ = tl.zeros((BLOCK_M,), dtype=tl.float32)
    sumsq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for start in range(0, M, BLOCK_M):
        idx = start + offs_m
        mask = idx < M
        v = tl.load(x_ptr + idx * N + col, mask=mask, other=0.0)
        sum_ += tl.where(mask, v, 0.0)
        sumsq += tl.where(mask, v * v, 0.0)
    s = tl.sum(sum_, axis=0)
    ss = tl.sum(sumsq, axis=0)
    mean = s / M
    var = ss / M - mean * mean
    tl.store(mean_ptr + col, mean)
    tl.store(var_ptr + col, var)


@triton.jit
def fused_affine_softmax_kernel(
    x_ptr, a_ptr, b_ptr, out_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * N
    out_row_ptr = out_ptr + row * N

    NEG_INF = float('-inf')

    # online softmax: single pass for max & sum
    m_run = NEG_INF
    s_run = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row_ptr + offs, mask=mask, other=0.0)
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        y = x * a + b
        y = tl.where(mask, y, NEG_INF)
        block_max = tl.max(y, axis=0)
        new_m = tl.maximum(m_run, block_max)
        s_run = s_run * tl.exp(m_run - new_m) + tl.sum(tl.exp(y - new_m), axis=0)
        m_run = new_m

    # second pass: write normalized values
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row_ptr + offs, mask=mask, other=0.0)
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        y = x * a + b
        e = tl.exp(y - m_run) / s_run
        tl.store(out_row_ptr + offs, e, mask=mask)


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

    def _build_scale_shift(self, mean, var, weight_bn, bias_bn, scale, bias_lin):
        # y_pre_softmax = scale * (bn_weight * (x - mean)/sqrt(var+eps) + bn_bias)
        # where x = lin_out
        # = lin_out * (scale * bn_weight / sqrt(var+eps)) + (scale * (bn_bias - bn_weight*mean/sqrt(var+eps)))
        inv = torch.rsqrt(var + self.bn_eps)
        a = scale * weight_bn * inv  # [N]
        b = scale * (bias_bn - weight_bn * mean * inv)  # [N]
        # The kernel adds linear bias separately and then applies a*x + b.
        # We need: a * (lin_out_with_bias) + b
        # Already lin_out_with_bias is computed (acc + bias). So fine.
        return a.contiguous(), b.contiguous()

    def forward(self, x):
        x = x.contiguous().cuda()
        M, K = x.shape
        N = self.out_features

        weight = self.gemm.weight  # [N, K]
        bias = self.gemm.bias      # [N]

        # GEMM + bias via custom kernel
        lin_out = torch.empty((M, N), device=x.device, dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
        gemm_bias_kernel[grid](
            x, weight, bias, lin_out,
            M, N, K,
            x.stride(0), x.stride(1),
            weight.stride(1), weight.stride(0),
        )

        if self.training:
            # compute batch mean/var per column via Triton kernel
            mean = torch.empty(N, device=x.device, dtype=torch.float32)
            var = torch.empty(N, device=x.device, dtype=torch.float32)
            col_mean_var_kernel[(N,)](lin_out, mean, var, M, N, BLOCK_M=1024, num_warps=8)

            with torch.no_grad():
                # unbiased var for running_var
                unbiased_var = var * (M / (M - 1)) if M > 1 else var
                self.bn.running_mean.mul_(1 - self.bn_momentum).add_(mean, alpha=self.bn_momentum)
                self.bn.running_var.mul_(1 - self.bn_momentum).add_(unbiased_var, alpha=self.bn_momentum)
                self.bn.num_batches_tracked.add_(1)
        else:
            mean = self.bn.running_mean
            var = self.bn.running_var

        a, b = self._build_scale_shift(mean, var, self.bn.weight, self.bn.bias, self.scale, bias)

        out = torch.empty_like(lin_out)
        BLOCK_N = 2048
        fused_affine_softmax_kernel[(M,)](lin_out, a, b, out, M, N, BLOCK_N=BLOCK_N, num_warps=8, num_stages=2)
        return out