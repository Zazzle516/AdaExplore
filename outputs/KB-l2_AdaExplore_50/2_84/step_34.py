import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
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

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n)
    acc += bias[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc)


@triton.jit
def col_stats_kernel(
    x_ptr, mean_ptr, var_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)  # over N columns
    col = pid
    if col < N:
        offs = tl.arange(0, BLOCK_M)
        s = 0.0
        sq = 0.0
        for m_start in range(0, M, BLOCK_M):
            m_offs = m_start + offs
            mask = m_offs < M
            x = tl.load(x_ptr + m_offs * N + col, mask=mask, other=0.0)
            s += tl.sum(x, axis=0)
            sq += tl.sum(x * x, axis=0)
        mean = s / M
        var = sq / M - mean * mean
        tl.store(mean_ptr + col, mean)
        tl.store(var_ptr + col, var)


@triton.jit
def affine_softmax_kernel(
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
    max_val = tl.max(v, axis=0)
    e = tl.exp(v - max_val)
    e = tl.where(mask, e, 0.0)
    sum_val = tl.sum(e, axis=0)
    out = e / sum_val
    tl.store(out_row_ptr + offs, out, mask=mask)


@triton.jit
def softmax_only_kernel(
    x_ptr, out_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * N
    out_row_ptr = out_ptr + row * N

    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    v = tl.load(x_row_ptr + offs, mask=mask, other=-float('inf'))
    max_val = tl.max(v, axis=0)
    e = tl.exp(v - max_val)
    e = tl.where(mask, e, 0.0)
    sum_val = tl.sum(e, axis=0)
    out = e / sum_val
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
        self._cached_a = None
        self._cached_b = None

    def train(self, mode=True):
        self._cached_a = None
        self._cached_b = None
        return super().train(mode)

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

        # Step 1: GEMM + bias -> lin_out
        lin_out = torch.empty((M, N), device=x.device, dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
        gemm_bias_kernel[grid](
            x, weight, bias, lin_out,
            M, N, K,
            x.stride(0), x.stride(1),
            weight.stride(1), weight.stride(0),
        )

        if self.training:
            # Step 2: compute per-column mean/var
            mean = torch.empty(N, device=x.device, dtype=torch.float32)
            var = torch.empty(N, device=x.device, dtype=torch.float32)
            BLOCK_M_STAT = 1024
            col_stats_kernel[(N,)](
                lin_out, mean, var,
                M, N,
                BLOCK_M=BLOCK_M_STAT,
                num_warps=4,
            )
            # update running stats
            with torch.no_grad():
                # var (unbiased) = var * M/(M-1)
                unbiased_var = var * (M / (M - 1)) if M > 1 else var
                self.bn.running_mean.mul_(1 - self.bn_momentum).add_(mean, alpha=self.bn_momentum)
                self.bn.running_var.mul_(1 - self.bn_momentum).add_(unbiased_var, alpha=self.bn_momentum)
                self.bn.num_batches_tracked.add_(1)
        else:
            mean = self.bn.running_mean
            var = self.bn.running_var

        if self.training:
            a, b = self._build_ab(mean, var, self.bn.weight, self.bn.bias, self.scale)
        else:
            # cache a, b in eval mode
            if (not hasattr(self, '_cached_a')) or self._cached_a is None:
                a, b = self._build_ab(mean, var, self.bn.weight, self.bn.bias, self.scale)
                self._cached_a = a
                self._cached_b = b
            a = self._cached_a
            b = self._cached_b

        # Step 3: fused affine + softmax (single-pass, full row)
        out = torch.empty_like(lin_out)
        # N=8192 in our setting; pick BLOCK_N >= N (power of 2)
        BLOCK_N = triton.next_power_of_2(N)
        affine_softmax_kernel[(M,)](
            lin_out, a, b, out,
            M, N,
            BLOCK_N=BLOCK_N,
            num_warps=8,
            num_stages=2,
        )
        return out