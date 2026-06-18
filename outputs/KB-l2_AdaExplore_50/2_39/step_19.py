import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_scale_kernel(
    A_ptr, B_ptr, bias_ptr, scale_ptr, C_ptr,
    sum_ptr, sumsq_ptr,
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
        a = tl.load(a_ptrs, mask=(offs_k[None, :] < k_remain) & mask_m[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remain) & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)
    acc = (acc + bias[None, :]) * scale[None, :]

    mask = mask_m[:, None] & mask_n[None, :]
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask)

    # Compute partial column sum and sumsq for this tile
    acc_masked = tl.where(mask, acc, 0.0)
    col_sum = tl.sum(acc_masked, axis=0)
    col_sumsq = tl.sum(acc_masked * acc_masked, axis=0)
    tl.atomic_add(sum_ptr + offs_n, col_sum, mask=mask_n)
    tl.atomic_add(sumsq_ptr + offs_n, col_sumsq, mask=mask_n)


@triton.jit
def bn_apply_kernel(
    X_ptr, Y_ptr,
    sum_ptr, sumsq_ptr,
    bn_weight_ptr, bn_bias_ptr,
    M, N, eps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    s = tl.load(sum_ptr + offs_n, mask=mask_n, other=0.0)
    sq = tl.load(sumsq_ptr + offs_n, mask=mask_n, other=0.0)
    mean = s / M
    var = sq / M - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(bn_weight_ptr + offs_n, mask=mask_n, other=0.0)
    b = tl.load(bn_bias_ptr + offs_n, mask=mask_n, other=0.0)
    scale = w * invstd
    shift = b - mean * scale

    x_ptrs = X_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = mask_m[:, None] & mask_n[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    y = x * scale[None, :] + shift[None, :]
    y_ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptrs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self.eps = eps
        self.momentum = momentum
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous()
        M, K = x.shape
        N = self.out_features
        W = self.gemm.weight  # [N, K]
        B = self.gemm.bias    # [N]
        # B layout for matmul: K x N, use weight transposed
        Wt = W.t().contiguous()

        out = torch.empty((M, N), device=x.device, dtype=torch.float32)
        col_sum = torch.zeros((N,), device=x.device, dtype=torch.float32)
        col_sumsq = torch.zeros((N,), device=x.device, dtype=torch.float32)

        scale_flat = self.scale.reshape(-1).contiguous()

        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
        gemm_scale_kernel[grid](
            x, Wt, B, scale_flat, out,
            col_sum, col_sumsq,
            M, N, K,
            x.stride(0), x.stride(1),
            Wt.stride(0), Wt.stride(1),
            out.stride(0), out.stride(1),
        )

        if self.training:
            mean = col_sum / M
            var = col_sumsq / M - mean * mean
            with torch.no_grad():
                self.bn.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
                # unbiased var for running stats
                unbiased_var = var * (M / max(M - 1, 1))
                self.bn.running_var.mul_(1 - self.momentum).add_(unbiased_var, alpha=self.momentum)

            y = torch.empty_like(out)
            BLOCK_M = 64
            BLOCK_N = 128
            grid2 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            bn_apply_kernel[grid2](
                out, y, col_sum, col_sumsq,
                self.bn.weight, self.bn.bias,
                M, N, self.eps,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            )
            return y
        else:
            # Use running stats
            invstd = 1.0 / torch.sqrt(self.bn.running_var + self.eps)
            scale_b = self.bn.weight * invstd
            shift_b = self.bn.bias - self.bn.running_mean * scale_b
            return out * scale_b.unsqueeze(0) + shift_b.unsqueeze(0)