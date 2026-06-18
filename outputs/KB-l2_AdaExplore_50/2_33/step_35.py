import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_scale_fused_kernel(
    A_ptr, B_ptr, bias_ptr, s_ptr, t_ptr, C_ptr,
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
    s = tl.load(s_ptr + offs_n, mask=mask_n, other=0.0)
    t = tl.load(t_ptr + offs_n, mask=mask_n, other=0.0)
    # acc = ((acc + bias) * scale) * s + t
    # where s = bn_w / sqrt(var+eps), t = bn_b - mean * s
    # scale fold: acc_sc = (acc + bias) * scale; out = acc_sc * s + t
    # We pre-fold: s_eff = scale * s, t_eff = bias*scale*s + t = bias*s_eff + t
    out = acc * s[None, :] + t[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, out, mask=mask)


@triton.jit
def bn_stats_kernel(
    X_ptr,
    mean_ptr, var_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)

    sum_x = tl.zeros((), dtype=tl.float32)
    sum_x2 = tl.zeros((), dtype=tl.float32)

    for m_start in range(0, M, BLOCK_M):
        m_idx = m_start + offs_m
        mask = m_idx < M
        x = tl.load(X_ptr + m_idx * N + pid_n, mask=mask, other=0.0)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    mean = sum_x / M
    var = sum_x2 / M - mean * mean
    tl.store(mean_ptr + pid_n, mean)
    tl.store(var_ptr + pid_n, var)


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
        gbias = self.gemm.bias  # (N,)
        scale = self.scale  # (N,)

        if not self.training:
            # Eval: fold BN with running stats
            inv_std = torch.rsqrt(self.bn.running_var + self.eps)
            s_bn = self.bn.weight * inv_std  # (N,)
            t_bn = self.bn.bias - self.bn.running_mean * s_bn  # (N,)
            # combined: out = (gemm + gbias) * scale * s_bn + t_bn
            s_eff = scale * s_bn
            t_eff = gbias * s_eff + t_bn

            # but gemm_scale_fused_kernel expects: out = acc * s + t, where acc = A @ B
            # so we use s = s_eff, t = t_eff (no separate bias add)
            # adjust: pass bias=zeros, s=s_eff, t=t_eff; or just inline
            out = torch.empty((M, N), device=x.device, dtype=x.dtype)
            zero_bias = torch.zeros_like(gbias)
            # Reuse kernel: acc + bias=0, so acc * s + t format won't work directly.
            # Let me make the kernel compute: out = acc * s + t where t already absorbs bias contribution.
            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
            gemm_scale_fused_kernel[grid](
                x, W, zero_bias, s_eff, t_eff, out,
                M, N, K,
                x.stride(0), x.stride(1),
                W.stride(1), W.stride(0),
                out.stride(0), out.stride(1),
            )
            return out
        else:
            # Training: do GEMM+scale, then compute BN stats and apply
            # Compute GEMM with bias and scale fused. Use kernel with s=scale, t=bias*scale
            s_pre = scale
            t_pre = gbias * scale
            pre = torch.empty((M, N), device=x.device, dtype=x.dtype)
            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
            gemm_scale_fused_kernel[grid](
                x, W, gbias, s_pre, t_pre, pre,
                M, N, K,
                x.stride(0), x.stride(1),
                W.stride(1), W.stride(0),
                pre.stride(0), pre.stride(1),
            )

            # Compute mean/var per column
            mean = torch.empty((N,), device=x.device, dtype=torch.float32)
            var = torch.empty((N,), device=x.device, dtype=torch.float32)
            BLOCK_M_STATS = 256
            bn_stats_kernel[(N,)](
                pre, mean, var, M, N, BLOCK_M=BLOCK_M_STATS,
            )

            # Update running stats
            with torch.no_grad():
                unbiased_var = var * (M / max(M - 1, 1))
                self.bn.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
                self.bn.running_var.mul_(1 - self.momentum).add_(unbiased_var, alpha=self.momentum)

            inv_std = torch.rsqrt(var + self.eps)
            s_bn = self.bn.weight * inv_std
            t_bn = self.bn.bias - mean * s_bn
            out = pre * s_bn + t_bn
            return out