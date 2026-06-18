import torch
import torch.nn as nn
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_scale_bias_kernel(
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
        k_remain = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remain), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remain) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)
    acc = (acc + bias[None, :]) * scale[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def bn_stats_kernel(
    X_ptr, mean_ptr, invstd_ptr,
    M, N, eps,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)  # column index
    offs_m = tl.arange(0, BLOCK_M)
    mask = offs_m < M

    # First pass: mean
    x = tl.load(X_ptr + offs_m * N + pid, mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(x, axis=0)
    mean = s / M
    # Second pass: variance via sum of squared deviations (numerically stable)
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / M
    invstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + pid, mean)
    tl.store(invstd_ptr + pid, invstd)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256}, num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def bn_apply_kernel(
    X_ptr, Y_ptr, mean_ptr, invstd_ptr, weight_ptr, bias_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
</old_str_3>

<new_str_3>
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256}, num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def bn_apply_kernel(
    X_ptr, Y_ptr, mean_ptr, invstd_ptr, weight_ptr, bias_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    mean = tl.load(mean_ptr + offs_n, mask=mask_n, other=0.0)
    invstd = tl.load(invstd_ptr + offs_n, mask=mask_n, other=0.0)
    w = tl.load(weight_ptr + offs_n, mask=mask_n, other=0.0)
    b = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    
    ptrs = X_ptr + offs_m[:, None] * N + offs_n[None, :]
    x = tl.load(ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    y = (x - mean[None, :]) * invstd[None, :] * w[None, :] + b[None, :]
    out_ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, y, mask=mask_m[:, None] & mask_n[None, :])


def gemm_scale_bias(x, weight, bias, scale):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_scale_bias_kernel[grid](
        x, weight, bias, scale, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        out.stride(0), out.stride(1),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self.eps = eps
        self.momentum = momentum
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous()
        # GEMM + bias + scale fused
        z = gemm_scale_bias(x, self.gemm.weight, self.gemm.bias, self.scale)
        
        M, N = z.shape
        if self.training:
            # compute mean/var
            mean = torch.empty(N, device=z.device, dtype=torch.float32)
            invstd = torch.empty(N, device=z.device, dtype=torch.float32)
            # M is padded up to next pow2 in BLOCK_M; choose 1024 (matches batch_size)
            BLOCK_M = triton.next_power_of_2(M)
            bn_stats_kernel[(N,)](z, mean, invstd, M, N, self.eps, BLOCK_M=BLOCK_M, num_warps=4)
            
            # update running stats
            with torch.no_grad():
                var = (1.0 / (invstd * invstd)) - self.eps
                # unbiased var for running
                unbiased_var = var * M / max(M - 1, 1)
                self.bn.running_mean.mul_(1 - self.momentum).add_(mean * self.momentum)
                self.bn.running_var.mul_(1 - self.momentum).add_(unbiased_var * self.momentum)
                self.bn.num_batches_tracked.add_(1)
            
            y = torch.empty_like(z)
            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
            bn_apply_kernel[grid](z, y, mean, invstd, self.bn.weight, self.bn.bias, M, N)
            return y
        else:
            mean = self.bn.running_mean
            invstd = 1.0 / torch.sqrt(self.bn.running_var + self.eps)
            y = torch.empty_like(z)
            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
            bn_apply_kernel[grid](z, y, mean, invstd, self.bn.weight, self.bn.bias, M, N)
            return y