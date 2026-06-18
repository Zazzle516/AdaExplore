import torch
import torch.nn as nn
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]


@triton.jit
def bn_stats_kernel(
    Y_ptr, MEAN_ptr, VAR_ptr,
    M, N,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # one program per BLOCK_N columns; reduce over M
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    sum_x = tl.zeros((BLOCK_N,), dtype=tl.float32)
    sum_x2 = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for m_start in range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        vals = tl.load(ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        sum_x += tl.sum(vals, axis=0)
        sum_x2 += tl.sum(vals * vals, axis=0)

    inv_m = 1.0 / M
    mean = sum_x * inv_m
    var = sum_x2 * inv_m - mean * mean
    tl.store(MEAN_ptr + offs_n, mean, mask=mask_n)
    tl.store(VAR_ptr + offs_n, var, mask=mask_n)


@triton.jit
def bn_apply_kernel(
    Y_ptr, OUT_ptr, SCALE_ptr, SHIFT_ptr,
    M, N,
    stride_ym, stride_yn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y = tl.load(y_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

    scale = tl.load(SCALE_ptr + offs_n, mask=mask_n, other=0.0)
    shift = tl.load(SHIFT_ptr + offs_n, mask=mask_n, other=0.0)

    z = y * scale[None, :] + shift[None, :]
    out = z * tl.sigmoid(z)

    out_ptrs = OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def fused_linear_bn_swish_kernel(
    A_ptr, B_ptr, bias_lin_ptr, scale_ptr, shift_ptr,
    OUT_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    INV_DIV: tl.constexpr,
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
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add linear bias (shape N)
    bias_lin = tl.load(bias_lin_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias_lin[None, :]

    # Apply BN affine fold: y = acc * scale + shift  (where scale/shift already contain extra bias and division)
    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)
    shift = tl.load(shift_ptr + offs_n, mask=mask_n, other=0.0)
    y = acc * scale[None, :] + shift[None, :]

    # Swish: y * sigmoid(y)
    out = y * tl.sigmoid(y)

    out_ptrs = OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask_m = offs_m < M
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = float(divide_value)
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum

    def _compute_fold(self, mean, var):
        inv_std = 1.0 / torch.sqrt(var + self.bn_eps)
        gamma = self.bn.weight
        beta = self.bn.bias
        scale = (gamma * inv_std) / self.divide_value
        extra = self.bias
        shift = (beta - mean * gamma * inv_std) / self.divide_value + extra / self.divide_value
        if shift.dim() == 0 or shift.numel() == 1:
            shift = shift.expand(self.out_features).contiguous()
        return scale.contiguous(), shift.contiguous()

    def forward(self, x):
        x = x.cuda().contiguous()
        if x.dtype != torch.float32:
            x = x.float()

        W = self.matmul.weight  # (out, in)
        b_lin = self.matmul.bias  # (out,)

        M, K = x.shape
        N = self.out_features

        if self.training:
            # Use cuBLAS for the matmul (very fast), then fused stats + apply
            # y = x @ W^T + b_lin
            y = torch.addmm(b_lin, x, W.t())  # (M, N)

            # Compute per-column mean/var via Triton
            mean = torch.empty(N, device=x.device, dtype=torch.float32)
            var = torch.empty(N, device=x.device, dtype=torch.float32)

            BLOCK_N_STATS = 128
            BLOCK_M_STATS = 128
            grid_stats = (triton.cdiv(N, BLOCK_N_STATS),)
            bn_stats_kernel[grid_stats](
                y, mean, var,
                M, N,
                y.stride(0), y.stride(1),
                BLOCK_M=BLOCK_M_STATS, BLOCK_N=BLOCK_N_STATS,
            )

            # Update running stats: unbiased variance for running_var
            with torch.no_grad():
                m = self.bn_momentum
                self.bn.running_mean.mul_(1 - m).add_(mean, alpha=m)
                # unbiased correction: var * M / (M-1)
                unbiased = var * (M / max(M - 1, 1))
                self.bn.running_var.mul_(1 - m).add_(unbiased, alpha=m)
                self.bn.num_batches_tracked.add_(1)

            # Compute fused scale/shift using batch stats
            scale, shift = self._compute_fold(mean, var)

            out = torch.empty_like(y)
            BLOCK_M_APP = 64
            BLOCK_N_APP = 256
            grid_app = (triton.cdiv(M, BLOCK_M_APP), triton.cdiv(N, BLOCK_N_APP))
            bn_apply_kernel[grid_app](
                y, out, scale, shift,
                M, N,
                y.stride(0), y.stride(1),
                out.stride(0), out.stride(1),
                BLOCK_M=BLOCK_M_APP, BLOCK_N=BLOCK_N_APP,
                num_warps=8, num_stages=3,
            )
            return out
        else:
            mean = self.bn.running_mean
            var = self.bn.running_var
            scale, shift = self._compute_fold(mean, var)

            out = torch.empty((M, N), device=x.device, dtype=torch.float32)

            grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
            fused_linear_bn_swish_kernel[grid](
                x, W, b_lin, scale, shift,
                out,
                M, N, K,
                x.stride(0), x.stride(1),
                1, W.stride(0),
                out.stride(0), out.stride(1),
                INV_DIV=1.0 / self.divide_value,
            )
            return out