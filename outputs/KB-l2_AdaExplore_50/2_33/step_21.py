import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------- GEMM with fused bias + scale, also accumulating column sums and sumsq ----------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=2, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_scale_stats_kernel(
    A_ptr, B_ptr, bias_ptr, scale_ptr,
    Y_ptr, sum_ptr, sumsq_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        a_mask = mask_m[:, None] & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & mask_n[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # bias + scale
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)
    acc = (acc + bias[None, :]) * scale[None, :]

    full_mask = mask_m[:, None] & mask_n[None, :]
    # store output
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=full_mask)

    # column reductions across this tile's M rows
    acc_masked = tl.where(full_mask, acc, 0.0)
    col_sum = tl.sum(acc_masked, axis=0)
    col_sumsq = tl.sum(acc_masked * acc_masked, axis=0)
    tl.atomic_add(sum_ptr + offs_n, col_sum, mask=mask_n)
    tl.atomic_add(sumsq_ptr + offs_n, col_sumsq, mask=mask_n)


# ---------- Finalize mean/invstd and update running stats ----------
@triton.jit
def finalize_stats_kernel(
    sum_ptr, sumsq_ptr,
    mean_ptr, invstd_ptr,
    running_mean_ptr, running_var_ptr,
    N, M, eps, momentum,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    s = tl.load(sum_ptr + offs, mask=mask, other=0.0)
    sq = tl.load(sumsq_ptr + offs, mask=mask, other=0.0)
    m_f = M.to(tl.float32)
    mean = s / m_f
    var_biased = sq / m_f - mean * mean
    # unbiased var for running stats
    var_unbiased = var_biased * (m_f / (m_f - 1.0))
    invstd = 1.0 / tl.sqrt(var_biased + eps)
    tl.store(mean_ptr + offs, mean, mask=mask)
    tl.store(invstd_ptr + offs, invstd, mask=mask)

    rm = tl.load(running_mean_ptr + offs, mask=mask, other=0.0)
    rv = tl.load(running_var_ptr + offs, mask=mask, other=0.0)
    rm_new = rm * (1.0 - momentum) + mean * momentum
    rv_new = rv * (1.0 - momentum) + var_unbiased * momentum
    tl.store(running_mean_ptr + offs, rm_new, mask=mask)
    tl.store(running_var_ptr + offs, rv_new, mask=mask)


# ---------- Apply normalization with weight/bias ----------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256}, num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def apply_bn_kernel(
    Y_ptr, mean_ptr, invstd_ptr, w_ptr, b_ptr,
    Out_ptr,
    M, N,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y = tl.load(ptrs, mask=mask, other=0.0)

    mean = tl.load(mean_ptr + offs_n, mask=mask_n, other=0.0)
    invstd = tl.load(invstd_ptr + offs_n, mask=mask_n, other=0.0)
    w = tl.load(w_ptr + offs_n, mask=mask_n, other=0.0)
    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)

    out = (y - mean[None, :]) * invstd[None, :] * w[None, :] + b[None, :]
    out_ptrs = Out_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(out_ptrs, out, mask=mask)


# ---------- Eval-mode fused apply ----------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def gemm_scale_bn_eval_kernel(
    A_ptr, B_ptr, bias_ptr, scale_ptr,
    rm_ptr, rv_ptr, w_ptr, bnb_ptr,
    Out_ptr,
    M, N, K, eps,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        a = tl.load(a_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)
    rm = tl.load(rm_ptr + offs_n, mask=mask_n, other=0.0)
    rv = tl.load(rv_ptr + offs_n, mask=mask_n, other=0.0)
    w = tl.load(w_ptr + offs_n, mask=mask_n, other=0.0)
    bnb = tl.load(bnb_ptr + offs_n, mask=mask_n, other=0.0)
    invstd = 1.0 / tl.sqrt(rv + eps)
    scaled = (acc + bias[None, :]) * scale[None, :]
    out = (scaled - rm[None, :]) * invstd[None, :] * w[None, :] + bnb[None, :]
    full_mask = mask_m[:, None] & mask_n[None, :]
    out_ptrs = Out_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(out_ptrs, out, mask=full_mask)


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
        x = x.contiguous().cuda()
        M, K = x.shape
        N = self.out_features

        W = self.gemm.weight  # [N, K]
        bias = self.gemm.bias  # [N]
        scale = self.scale.view(-1)  # [N]
        # B in kernel is W.T: shape [K, N]
        # We avoid creating it; pass strides instead.

        if self.training:
            Y = torch.empty((M, N), device=x.device, dtype=torch.float32)
            sum_buf = torch.zeros((N,), device=x.device, dtype=torch.float32)
            sumsq_buf = torch.zeros((N,), device=x.device, dtype=torch.float32)

            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
            gemm_scale_stats_kernel[grid](
                x, W, bias, scale,
                Y, sum_buf, sumsq_buf,
                M, N, K,
                x.stride(0), x.stride(1),
                W.stride(1), W.stride(0),  # B = W.T → stride_bk = W.stride(1), stride_bn = W.stride(0)
                Y.stride(0), Y.stride(1),
            )

            mean = torch.empty((N,), device=x.device, dtype=torch.float32)
            invstd = torch.empty((N,), device=x.device, dtype=torch.float32)
            BLOCK = 256
            grid2 = (triton.cdiv(N, BLOCK),)
            finalize_stats_kernel[grid2](
                sum_buf, sumsq_buf,
                mean, invstd,
                self.bn.running_mean, self.bn.running_var,
                N, M, self.eps, self.momentum,
                BLOCK=BLOCK,
            )
            self.bn.num_batches_tracked += 1

            out = torch.empty_like(Y)
            grid3 = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
            apply_bn_kernel[grid3](
                Y, mean, invstd, self.bn.weight, self.bn.bias,
                out,
                M, N,
                Y.stride(0), Y.stride(1),
            )
            return out
        else:
            out = torch.empty((M, N), device=x.device, dtype=torch.float32)
            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
            gemm_scale_bn_eval_kernel[grid](
                x, W, bias, scale,
                self.bn.running_mean, self.bn.running_var, self.bn.weight, self.bn.bias,
                out,
                M, N, K, self.eps,
                x.stride(0), x.stride(1),
                W.stride(1), W.stride(0),
                out.stride(0), out.stride(1),
            )
            return out