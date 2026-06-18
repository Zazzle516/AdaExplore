import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_bias_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
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
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias = tl.load(Bias_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def instnorm_residual_kernel(
    X_ptr, Y_ptr, Out_ptr,
    M, N,
    eps,
    BLOCK_N: tl.constexpr,
    TILES: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return
    x_row = X_ptr + row * N
    y_row = Y_ptr + row * N
    o_row = Out_ptr + row * N

    # Pass 1: mean
    sum_x = 0.0
    for t in range(0, TILES):
        offs = t * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.sum(x, axis=0)
    mean = sum_x / N

    # Pass 2: centered variance
    sum_d2 = 0.0
    for t in range(0, TILES):
        offs = t * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        d = tl.where(mask, x - mean, 0.0)
        sum_d2 += tl.sum(d * d, axis=0)
    var = sum_d2 / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 3: normalize, add residual, multiply
    for t in range(0, TILES):
        offs = t * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_row + offs, mask=mask, other=0.0).to(tl.float32)
        xn = (x - mean) * rstd
        out = (xn + y) * y
        tl.store(o_row + offs, out, mask=mask)


def triton_gemm_bias(x, w_t, b):
    # w_t is (K, N) contiguous
    M, K = x.shape
    K2, N = w_t.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_bias_kernel[grid](
        x, w_t, b, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


def triton_instnorm_residual(x, y, eps):
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_N = 1024
    TILES = (N + BLOCK_N - 1) // BLOCK_N
    grid = (M,)
    instnorm_residual_kernel[grid](
        x, y, out,
        M, N,
        eps,
        BLOCK_N=BLOCK_N,
        TILES=TILES,
        num_warps=8,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)
        self.eps = eps
        self._w_t_cache = None

    def _get_w_t(self):
        w = self.bmm.weight  # (N, K)
        if (self._w_t_cache is None
                or self._w_t_cache.device != w.device
                or self._w_t_cache.dtype != w.dtype
                or self._w_t_cache.shape[0] != w.shape[1]
                or self._w_t_cache.shape[1] != w.shape[0]):
            self._w_t_cache = w.t().contiguous()
        return self._w_t_cache

    def forward(self, x, y):
        x = x.contiguous()
        y = y.contiguous()
        w_t = self._get_w_t()
        b = self.bmm.bias.contiguous()
        z = triton_gemm_bias(x, w_t, b)
        out = triton_instnorm_residual(z, y, self.eps)
        return out