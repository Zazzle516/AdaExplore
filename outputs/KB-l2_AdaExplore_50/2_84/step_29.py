import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_affine_kernel(
    a_ptr, b_ptr, bias_ptr, scale_shift_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

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

    # bias
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    # apply scale-shift fold: y = acc * scale_n + shift_n
    # We pack [scale, shift] as 2*N tensor (scale in first N, shift in second N)
    scale = tl.load(scale_shift_ptr + offs_n, mask=mask_n, other=0.0)
    shift = tl.load(scale_shift_ptr + N + offs_n, mask=mask_n, other=0.0)
    acc = acc * scale[None, :] + shift[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def softmax_kernel(
    x_ptr, out_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * N
    out_row_ptr = out_ptr + row * N

    # First pass: compute max
    max_val = -float('inf')
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row_ptr + offs, mask=mask, other=-float('inf'))
        block_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Second pass: compute sum of exp
    sum_val = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row_ptr + offs, mask=mask, other=-float('inf'))
        e = tl.exp(x - max_val)
        e = tl.where(mask, e, 0.0)
        sum_val += tl.sum(e, axis=0)

    # Third pass: write normalized
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row_ptr + offs, mask=mask, other=-float('inf'))
        e = tl.exp(x - max_val) / sum_val
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

        # Compute mean/var at runtime (training mode by default for BN)
        # In training, BN uses batch statistics. We need to compute these.
        # Simplest approach: do gemm with PyTorch-equivalent stats computation.
        # However we want fused. Use two-pass: first compute lin output for stats.
        
        # For simplicity and correctness, do gemm via our kernel into a temp tensor first
        # without scale/shift, then compute BN stats, then apply scale/shift via a second pass.
        # But to keep performance: do gemm normally, compute stats, then fuse the rest.

        # Strategy: do GEMM+bias in one kernel (with scale=1, shift=0), then compute BN stats,
        # then a second elementwise+softmax fused pass.
        
        if self.training:
            # First do gemm
            lin_out = F.linear(x, weight, bias)  # fall back; or use kernel with identity scale
            # update running stats
            with torch.no_grad():
                batch_mean = lin_out.mean(dim=0)
                batch_var = lin_out.var(dim=0, unbiased=False)
                self.bn.running_mean.mul_(1 - self.bn_momentum).add_(batch_mean, alpha=self.bn_momentum)
                self.bn.running_var.mul_(1 - self.bn_momentum).add_(lin_out.var(dim=0, unbiased=True), alpha=self.bn_momentum)
                self.bn.num_batches_tracked.add_(1)
            mean = batch_mean
            var = batch_var
            # apply scale-shift
            a, b = self._build_scale_shift(mean, var, self.bn.weight, self.bn.bias, self.scale, bias)
            y = lin_out * a + b
        else:
            mean = self.bn.running_mean
            var = self.bn.running_var
            a, b = self._build_scale_shift(mean, var, self.bn.weight, self.bn.bias, self.scale, bias)
            # Use fused gemm+affine kernel
            scale_shift = torch.empty(2 * N, device=x.device, dtype=torch.float32)
            scale_shift[:N] = a
            scale_shift[N:] = b
            y = torch.empty((M, N), device=x.device, dtype=torch.float32)

            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
            gemm_affine_kernel[grid](
                x, weight, bias, scale_shift, y,
                M, N, K,
                x.stride(0), x.stride(1),
                weight.stride(1), weight.stride(0),
            )

        # Softmax along dim=1
        out = torch.empty_like(y)
        BLOCK_N = 1024
        softmax_kernel[(M,)](y, out, M, N, BLOCK_N=BLOCK_N, num_warps=8)
        return out