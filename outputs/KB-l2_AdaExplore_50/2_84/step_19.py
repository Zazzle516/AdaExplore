import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bn_scale_kernel(
    x_ptr, w_ptr, scale_shift_ptr, scale_mul_ptr,
    out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # Group-major ordering for better L2 reuse
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # W is stored as K x N (transposed once at init), stride_wk=N, stride_wn=1
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=offs_m[:, None] < M, other=0.0)
        w = tl.load(w_ptrs, mask=offs_n[None, :] < N, other=0.0)
        acc += tl.dot(x, w, allow_tf32=True)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

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
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + pid * stride_xm + offs, mask=mask, other=float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(out_ptr + pid * stride_om + offs, y, mask=mask)


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

        self._cached_mul = None
        self._cached_shift = None
        self._cached_wt = None  # K x N transposed weight

    def _compute_fused(self, dtype, device):
        W = self.gemm.weight
        b = self.gemm.bias
        rm = self.bn.running_mean
        rv = self.bn.running_var
        bn_w = self.bn.weight
        bn_b = self.bn.bias
        scale = self.scale

        inv_std = torch.rsqrt(rv + self.bn_eps)
        mul = (scale * inv_std * bn_w).contiguous().to(dtype)
        shift = (scale * ((b - rm) * inv_std * bn_w + bn_b)).contiguous().to(dtype)

        mul = mul.view(-1)
        shift = shift.view(-1)
        N = self.out_features
        if mul.numel() == 1:
            mul = mul.expand(N).contiguous()
        if shift.numel() == 1:
            shift = shift.expand(N).contiguous()
        return mul, shift

    def forward(self, x):
        x = x.contiguous()
        M, K = x.shape
        N = self.out_features

        if self.training:
            y = self.gemm(x)
            y = self.bn(y)
            y = self.scale * y
            y = self.softmax(y)
            return y

        if (self._cached_mul is None
                or self._cached_mul.device != x.device
                or self._cached_mul.dtype != x.dtype):
            mul, shift = self._compute_fused(x.dtype, x.device)
            self._cached_mul = mul
            self._cached_shift = shift
            # Transpose weight once: (N, K) -> (K, N) contiguous for contiguous K-axis loads
            self._cached_wt = self.gemm.weight.t().contiguous().to(x.dtype)

        mul = self._cached_mul
        shift = self._cached_shift
        Wt = self._cached_wt  # K x N

        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
        gemm_bn_scale_kernel[grid](
            x, Wt, shift, mul,
            out,
            M, N, K,
            x.stride(0), x.stride(1),
            Wt.stride(0), Wt.stride(1),
            out.stride(0), out.stride(1),
        )

        sm_out = torch.empty_like(out)
        # N=8192 fits in registers/SRAM, single-pass softmax
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 16 if BLOCK_N >= 8192 else (8 if BLOCK_N >= 2048 else 4)
        softmax_kernel[(M,)](
            out, sm_out,
            M, N,
            out.stride(0), sm_out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return sm_out