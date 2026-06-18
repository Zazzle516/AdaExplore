import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bn_scale_kernel(
    x_ptr, w_ptr, scale_shift_ptr, scale_mul_ptr,
    out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] < k_remaining), other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # apply scale_mul * (gemm_out + shift_eff), where shift_eff includes bias and bn affine
    mul = tl.load(scale_mul_ptr + offs_n, mask=offs_n < N, other=0.0)
    shift = tl.load(scale_shift_ptr + offs_n, mask=offs_n < N, other=0.0)

    out = acc * mul[None, :] + shift[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def softmax_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_xm, stride_om,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    # online / two-pass softmax across N
    # Pass 1: max
    max_val = -float('inf')
    for n_start in range(0, N, BLOCK_N):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + pid * stride_xm + offs, mask=mask, other=-float('inf'))
        block_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Pass 2: sum exp
    sum_exp = 0.0
    for n_start in range(0, N, BLOCK_N):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + pid * stride_xm + offs, mask=mask, other=-float('inf'))
        e = tl.exp(x - max_val)
        sum_exp += tl.sum(e, axis=0)

    inv = 1.0 / sum_exp

    # Pass 3: write
    for n_start in range(0, N, BLOCK_N):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + pid * stride_xm + offs, mask=mask, other=-float('inf'))
        e = tl.exp(x - max_val) * inv
        tl.store(out_ptr + pid * stride_om + offs, e, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        self.scale_shape = scale_shape

        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.contiguous()
        M, K = x.shape
        N = self.out_features

        if self.training:
            # fall back to standard path during training to keep BN stats updates
            y = self.gemm(x)
            y = self.bn(y)
            y = self.scale * y
            y = self.softmax(y)
            return y

        # Inference: fold bn affine + bias + scale into gemm epilogue
        W = self.gemm.weight  # (N, K)
        b = self.gemm.bias    # (N,)
        rm = self.bn.running_mean
        rv = self.bn.running_var
        bn_w = self.bn.weight
        bn_b = self.bn.bias
        scale = self.scale  # broadcastable

        inv_std = torch.rsqrt(rv + self.bn_eps)
        # bn affine: (y - rm) * inv_std * bn_w + bn_b
        # full epilogue: scale * (((gemm + b) - rm) * inv_std * bn_w + bn_b)
        # = gemm * (scale * inv_std * bn_w) + scale * ((b - rm) * inv_std * bn_w + bn_b)
        mul = (scale * inv_std * bn_w).contiguous().to(x.dtype)
        shift = (scale * ((b - rm) * inv_std * bn_w + bn_b)).contiguous().to(x.dtype)

        # ensure shape (N,)
        mul = mul.view(-1)
        shift = shift.view(-1)
        if mul.numel() == 1:
            mul = mul.expand(N).contiguous()
        if shift.numel() == 1:
            shift = shift.expand(N).contiguous()

        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
        gemm_bn_scale_kernel[grid](
            x, W, shift, mul,
            out,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            out.stride(0), out.stride(1),
        )

        # softmax along dim=1
        sm_out = torch.empty_like(out)
        BLOCK_N = 1024
        softmax_kernel[(M,)](
            out, sm_out,
            M, N,
            out.stride(0), sm_out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return sm_out