import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def fused_gemm_bn_kernel(
    x_ptr, w_ptr, A_ptr, Bp_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
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

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k0 in range(0, K, BLOCK_K):
        k_remaining = K - k0
        mask_k = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    A = tl.load(A_ptr + offs_n, mask=mask_n, other=0.0)
    Bp = tl.load(Bp_ptr + offs_n, mask=mask_n, other=0.0)

    out = acc * A[None, :] + Bp[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def softmax_kernel_full(
    x_ptr, out_ptr, n_cols,
    stride_x_row, stride_o_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_x_row
    o_row = out_ptr + row * stride_o_row

    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_row + offs, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(o_row + offs, y, mask=mask)


@triton.jit
def softmax_kernel_tiled(
    x_ptr, out_ptr, n_cols,
    stride_x_row, stride_o_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_x_row
    o_row = out_ptr + row * stride_o_row

    offs = tl.arange(0, BLOCK)
    max_val = -float('inf')
    for c0 in range(0, n_cols, BLOCK):
        cols = c0 + offs
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=-float('inf'))
        cur_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, cur_max)

    sum_val = 0.0
    for c0 in range(0, n_cols, BLOCK):
        cols = c0 + offs
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=-float('inf'))
        e = tl.exp(x - max_val)
        sum_val += tl.sum(tl.where(mask, e, 0.0), axis=0)

    inv_sum = 1.0 / sum_val

    for c0 in range(0, n_cols, BLOCK):
        cols = c0 + offs
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=-float('inf'))
        e = tl.exp(x - max_val) * inv_sum
        tl.store(o_row + cols, e, mask=mask)


def fused_gemm_bn(x, weight_t, A, Bp):
    # weight_t: [K, N] contiguous
    M, K = x.shape
    N = weight_t.shape[1]
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    fused_gemm_bn_kernel[grid](
        x, weight_t, A, Bp, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


def triton_softmax(x):
    M, N = x.shape
    out = torch.empty_like(x)
    # Choose BLOCK >= N if power of two and fits, else tiled.
    if N <= 8192:
        # Find next power of 2 >= N
        BLOCK = 1
        while BLOCK < N:
            BLOCK *= 2
        grid = (M,)
        nw = 16 if BLOCK >= 4096 else 8
        softmax_kernel_full[grid](
            x, out, N,
            x.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=nw, num_stages=1,
        )
    else:
        BLOCK = 2048
        grid = (M,)
        softmax_kernel_tiled[grid](
            x, out, N,
            x.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=8, num_stages=2,
        )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.softmax = nn.Softmax(dim=1)
        self.in_features = in_features
        self.out_features = out_features
        self._cached_A = None
        self._cached_Bp = None
        self._cached_Wt = None
        self._cache_key = None

    def _build_fused_params(self):
        bn_mean = self.bn.running_mean
        bn_var = self.bn.running_var
        bn_eps = self.bn.eps
        bn_w = self.bn.weight
        bn_b = self.bn.bias

        bn_scale = bn_w / torch.sqrt(bn_var + bn_eps)
        scale = self.scale
        A = (scale * bn_scale).contiguous()
        Bp = (scale * (bn_scale * (self.gemm.bias - bn_mean) + bn_b)).contiguous()
        return A, Bp

    def _get_cached(self):
        # Cache key based on tensor versions
        key = (
            self.bn.running_mean._version,
            self.bn.running_var._version,
            self.bn.weight._version,
            self.bn.bias._version,
            self.scale._version,
            self.gemm.weight._version,
            self.gemm.bias._version,
        )
        if self._cache_key != key or self._cached_Wt is None:
            A, Bp = self._build_fused_params()
            Wt = self.gemm.weight.t().contiguous()
            self._cached_A = A
            self._cached_Bp = Bp
            self._cached_Wt = Wt
            self._cache_key = key
        return self._cached_Wt, self._cached_A, self._cached_Bp

    def forward(self, x):
        x = x.cuda().contiguous()
        if self.training:
            y = self.gemm(x)
            y = self.bn(y)
            y = self.scale * y
            y = self.softmax(y)
            return y

        Wt, A, Bp = self._get_cached()
        y = fused_gemm_bn(x, Wt, A, Bp)
        y = triton_softmax(y)
        return y