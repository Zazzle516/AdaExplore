import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bn_scale_kernel(
    A_ptr, B_ptr, fused_w_ptr, fused_b_ptr, C_ptr,
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

    for k in range(0, K, BLOCK_K):
        mask_k = offs_k < (K - k)
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # load fused scale (per-N) and bias (per-N)
    w = tl.load(fused_w_ptr + offs_n, mask=mask_n, other=0.0)
    bias = tl.load(fused_b_ptr + offs_n, mask=mask_n, other=0.0)

    acc = acc * w[None, :] + bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def softmax_kernel(
    X_ptr, Y_ptr, M, N,
    stride_xm, stride_ym,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x_ptrs = X_ptr + row * stride_xm + offs
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))

    m = tl.max(x, axis=0)
    x = x - m
    e = tl.exp(x)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    y_ptrs = Y_ptr + row * stride_ym + offs
    tl.store(y_ptrs, y, mask=mask)


@triton.jit
def softmax_online_kernel(
    X_ptr, Y_ptr, M, N,
    stride_xm, stride_ym,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = X_ptr + row * stride_xm
    y_row = Y_ptr + row * stride_ym

    # Pass 1: online compute max and sum
    m = -float('inf')
    s = 0.0
    for n_start in range(0, N, BLOCK_N):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=-float('inf'))
        block_max = tl.max(x, axis=0)
        new_m = tl.maximum(m, block_max)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), axis=0)
        m = new_m

    # Pass 2: write normalized output
    inv_s = 1.0 / s
    for n_start in range(0, N, BLOCK_N):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=-float('inf'))
        e = tl.exp(x - m) * inv_s
        tl.store(y_row + offs, e, mask=mask)


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
        self.softmax = nn.Softmax(dim=1)
        self._wt_cache = None
        self._wt_version = -1

    def _get_wt(self):
        w = self.gemm.weight
        if self._wt_cache is None or self._wt_version != w._version or not self._wt_cache.is_cuda:
            self._wt_cache = w.t().contiguous().cuda() if not w.is_cuda else w.t().contiguous()
            self._wt_version = w._version
        return self._wt_cache

    def _compute_fused(self, device, dtype):
        # In eval mode, we can use running stats. In train mode, BN computes stats from batch.
        # For training mode, we need to fall back to the standard path or compute stats.
        # We'll always use the training-mode behavior matching: but here we use eval-mode fused if not training.
        eps = self.bn_eps
        if self.training:
            return None, None
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        bn_w = self.bn.weight
        bn_b = self.bn.bias
        scale = self.scale  # shape (1,) typically, broadcastable

        # output_pre_softmax = scale * (bn_w * (gemm_out - mean) / sqrt(var+eps) + bn_b)
        # = gemm_out * (scale * bn_w / sqrt(var+eps)) + scale * (bn_b - bn_w*mean/sqrt(var+eps))
        inv_std = torch.rsqrt(running_var + eps)
        fused_w = (scale * bn_w * inv_std).contiguous()  # shape (out_features,) after broadcast
        fused_b = (scale * (bn_b - bn_w * running_mean * inv_std)).contiguous()
        # Ensure shape (out_features,)
        if fused_w.dim() == 0:
            fused_w = fused_w.expand(self.out_features).contiguous()
        else:
            fused_w = fused_w.view(-1)
            if fused_w.numel() == 1:
                fused_w = fused_w.expand(self.out_features).contiguous()
        if fused_b.dim() == 0:
            fused_b = fused_b.expand(self.out_features).contiguous()
        else:
            fused_b = fused_b.view(-1)
            if fused_b.numel() == 1:
                fused_b = fused_b.expand(self.out_features).contiguous()

        # Now incorporate the linear bias: gemm output = x @ W^T + linear_bias
        # We compute (x @ W^T) in the kernel without bias, then apply: acc * fused_w + (linear_bias * fused_w + fused_b)
        linear_bias = self.gemm.bias
        final_b = (linear_bias * fused_w + fused_b).contiguous()

        return fused_w, final_b

    def forward(self, x):
        x = x.cuda() if not x.is_cuda else x
        x = x.contiguous()

        if self.training:
            # fallback to standard implementation
            y = self.gemm(x)
            y = self.bn(y)
            y = self.scale * y
            y = self.softmax(y)
            return y

        fused_w, fused_b = self._compute_fused(x.device, x.dtype)

        M, K = x.shape
        N = self.out_features
        Wt = self._get_wt()  # (in_features, out_features), contiguous
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
        gemm_bn_scale_kernel[grid](
            x, Wt, fused_w, fused_b, out,
            M, N, K,
            x.stride(0), x.stride(1),
            Wt.stride(0), Wt.stride(1),
            out.stride(0), out.stride(1),
        )

        # softmax
        y = torch.empty_like(out)
        BLOCK_N = 1
        while BLOCK_N < N:
            BLOCK_N *= 2
        if BLOCK_N <= 4096:
            num_warps = 8 if BLOCK_N >= 2048 else 4
            softmax_kernel[(M,)](
                out, y, M, N,
                out.stride(0), y.stride(0),
                BLOCK_N=BLOCK_N, num_warps=num_warps,
            )
            return y
        else:
            # online 2-pass softmax for wide rows
            softmax_online_kernel[(M,)](
                out, y, M, N,
                out.stride(0), y.stride(0),
                BLOCK_N=4096, num_warps=8, num_stages=2,
            )
            return y