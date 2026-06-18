import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_scale_bn_kernel(
    A_ptr, B_ptr, bias_ptr, scale_ptr, shift_ptr, out_ptr,
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

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k[None, :] < (K - k * BLOCK_K)
        a = tl.load(a_ptrs, mask=mask_m[:, None] & k_mask, other=0.0)
        k_mask_b = offs_k[:, None] < (K - k * BLOCK_K)
        b = tl.load(b_ptrs, mask=k_mask_b & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    # Apply scale * x then BN: y = (acc * scale - shift_part) * scale_part
    # We fold: after scale, BN computes (x - mean) / sqrt(var+eps) * gamma + beta
    # We precompute combined: out = acc * combined_scale + combined_shift
    # where combined_scale = scale * gamma / sqrt(var+eps)
    # and combined_shift = beta - mean * gamma / sqrt(var+eps)
    # But mean/var depend on (acc*scale) at runtime in training... 
    # Wait - this requires runtime BN stats. Need to handle separately.
    # Here we just do acc * scale (the scale param), then store, then BN externally.
    s = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc * s[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def gemm_scale(x, weight, bias, scale):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_scale_bn_kernel[grid](
        x, weight, bias, scale, scale, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # transpose
        out.stride(0), out.stride(1),
    )
    return out


@triton.jit
def bn_stats_kernel(
    x_ptr, mean_ptr, var_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    
    sum_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    sum2_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    for m_start in range(0, M, BLOCK_M):
        idx = m_start + offs_m
        mask = idx < M
        x = tl.load(x_ptr + idx * N + pid, mask=mask, other=0.0)
        xf = x.to(tl.float32)
        sum_acc += xf
        sum2_acc += xf * xf
    
    sum_x = tl.sum(sum_acc, axis=0)
    sum_x2 = tl.sum(sum2_acc, axis=0)
    
    mean = sum_x / M
    var = sum_x2 / M - mean * mean
    tl.store(mean_ptr + pid, mean)
    tl.store(var_ptr + pid, var)


@triton.jit
def bn_apply_kernel(
    x_ptr, out_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    M, N, eps,
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
    
    inv_std = 1.0 / tl.sqrt(var + eps)
    scale = gamma * inv_std
    shift = beta - mean * scale
    
    ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    x = tl.load(ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    y = x * scale[None, :] + shift[None, :]
    
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, y, mask=mask_m[:, None] & mask_n[None, :])


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
        x = x.contiguous().cuda()
        W = self.gemm.weight.contiguous()
        b = self.gemm.bias.contiguous()
        s = self.scale.contiguous()
        
        # Fused GEMM + scale
        y = gemm_scale(x, W, b, s)
        
        M, N = y.shape
        
        if self.training:
            # Compute mean/var via Triton
            mean = torch.empty(N, device=y.device, dtype=torch.float32)
            var = torch.empty(N, device=y.device, dtype=torch.float32)
            
            bn_stats_kernel[(N,)](y, mean, var, M, N, BLOCK_M=1024)
            
            # Update running stats
            with torch.no_grad():
                unbiased_var = var * (M / (M - 1)) if M > 1 else var
                self.bn.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
                self.bn.running_var.mul_(1 - self.momentum).add_(unbiased_var, alpha=self.momentum)
                self.bn.num_batches_tracked.add_(1)
        else:
            mean = self.bn.running_mean
            var = self.bn.running_var
        
        out = torch.empty_like(y)
        BLOCK_M = 64
        BLOCK_N = 128
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        bn_apply_kernel[grid](
            y, out, mean, var, self.bn.weight, self.bn.bias,
            M, N, self.eps,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )
        return out