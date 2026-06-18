import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=2, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_scale_kernel(
    A_ptr, B_ptr, Bias_ptr, Scale_ptr, C_ptr,
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

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_cn < N
    bias = tl.load(Bias_ptr + offs_cn, mask=n_mask, other=0.0)
    scale = tl.load(Scale_ptr + offs_cn, mask=n_mask, other=0.0)

    out = (acc + bias[None, :]) * scale[None, :]

    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, out, mask=c_mask)


@triton.jit
def bn_stats_kernel(
    X_ptr, Mean_ptr, InvStd_ptr,
    M, N, eps,
    stride_xm, stride_xn,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)  # column index
    if pid >= N:
        return
    offs_m = tl.arange(0, BLOCK_M)
    col_ptr = X_ptr + pid * stride_xn

    sum_x = tl.zeros((BLOCK_M,), dtype=tl.float32)
    sum_x2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        idx = m_start + offs_m
        mask = idx < M
        x = tl.load(col_ptr + idx * stride_xm, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.where(mask, x, 0.0)
        sum_x2 += tl.where(mask, x * x, 0.0)

    s = tl.sum(sum_x, axis=0)
    s2 = tl.sum(sum_x2, axis=0)
    mean = s / M
    var = s2 / M - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    tl.store(Mean_ptr + pid, mean)
    tl.store(InvStd_ptr + pid, inv_std)


@triton.jit
def bn_apply_kernel(
    X_ptr, Y_ptr, Mean_ptr, InvStd_ptr, Weight_ptr, Bias_ptr,
    M, N,
    stride_xm, stride_xn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    mean = tl.load(Mean_ptr + offs_n, mask=mask_n, other=0.0)
    inv_std = tl.load(InvStd_ptr + offs_n, mask=mask_n, other=0.0)
    w = tl.load(Weight_ptr + offs_n, mask=mask_n, other=0.0)
    b = tl.load(Bias_ptr + offs_n, mask=mask_n, other=0.0)

    ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    mask = mask_m[:, None] & mask_n[None, :]
    x = tl.load(ptrs, mask=mask, other=0.0)
    y = (x - mean[None, :]) * inv_std[None, :] * w[None, :] + b[None, :]
    tl.store(Y_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn, y, mask=mask)


def fused_gemm_scale(x, w, bias, scale):
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_scale_kernel[grid](
        x, w.t().contiguous() if not w.t().is_contiguous() else w.t(), bias, scale, out,
        M, N, K,
        x.stride(0), x.stride(1),
        1, N,  # using w transposed: shape (K, N) with stride_bk=N, stride_bn=1? Need consistent
        out.stride(0), out.stride(1),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self.eps = eps

    def _gemm_scale(self, x):
        M, K = x.shape
        N = self.out_features
        # B in (K, N) layout: weight is (N, K), so we use its transpose
        W = self.gemm.weight  # (N, K)
        B_t = W.t()  # (K, N), strides: stride_bk = 1, stride_bn = K
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
        gemm_scale_kernel[grid](
            x, B_t, self.gemm.bias, self.scale, out,
            M, N, K,
            x.stride(0), x.stride(1),
            B_t.stride(0), B_t.stride(1),
            out.stride(0), out.stride(1),
        )
        return out

    def forward(self, x):
        x = x.cuda().contiguous()
        if not self.gemm.weight.is_cuda:
            self.cuda()

        y = self._gemm_scale(x)  # (M, N)
        M, N = y.shape

        if self.training:
            # Compute stats on the fly
            mean = torch.empty(N, device=y.device, dtype=torch.float32)
            inv_std = torch.empty(N, device=y.device, dtype=torch.float32)
            BLOCK_M = 1024
            bn_stats_kernel[(N,)](
                y, mean, inv_std,
                M, N, self.eps,
                y.stride(0), y.stride(1),
                BLOCK_M=BLOCK_M,
            )
            # update running stats
            with torch.no_grad():
                batch_var_unbiased = None
                # Simple path: fall back to PyTorch BN update via direct compute
                var = (1.0 / (inv_std * inv_std)) - self.eps
                self.bn.running_mean.mul_(1 - self.bn.momentum).add_(mean * self.bn.momentum)
                if M > 1:
                    unbiased_var = var * (M / (M - 1))
                else:
                    unbiased_var = var
                self.bn.running_var.mul_(1 - self.bn.momentum).add_(unbiased_var * self.bn.momentum)
        else:
            mean = self.bn.running_mean
            inv_std = torch.rsqrt(self.bn.running_var + self.eps)

        out = torch.empty_like(y)
        BLOCK_M = 64
        BLOCK_N = 128
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        bn_apply_kernel[grid](
            y, out, mean, inv_std, self.bn.weight, self.bn.bias,
            M, N,
            y.stride(0), y.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )
        return out