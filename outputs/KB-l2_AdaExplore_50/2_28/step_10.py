import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_kernel(
    A_ptr, B_ptr, C_ptr, Bias_ptr,
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

    bias = tl.load(Bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
    acc += bias[None, :]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_linear(x, w_t, bias):
    # x: (M, K), w_t: (K, N), bias: (N,)
    M, K = x.shape
    K2, N = w_t.shape
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_bias_kernel[grid](
        x, w_t, out, bias,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


@triton.jit
def fused_norm_add_mul_kernel(
    x_ptr, y_ptr, out_ptr,
    N,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # InstanceNorm2d on (B, C, 1, 1): each (n,c) normalizes over 1 element
    # -> mean = x, var = 0, normed = (x - x) / sqrt(eps) = 0
    # -> out = (0 + y) * y = y * y
    # We still load x so the GEMM output is consumed.
    row = tl.program_id(0)
    x_row = x_ptr + row * N
    y_row = y_ptr + row * N
    o_row = out_ptr + row * N

    for off in range(0, N, BLOCK_N):
        offs = off + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_row + offs, mask=mask, other=0.0).to(tl.float32)
        normed = (x - x) * (1.0 / tl.sqrt(eps))
        out = (normed + y) * y
        tl.store(o_row + offs, out, mask=mask)


def fused_norm_add_mul(x, y, eps):
    x = x.contiguous()
    y = y.contiguous()
    B, N = x.shape
    out = torch.empty_like(x)
    BLOCK_N = 2048
    grid = (B,)
    fused_norm_add_mul_kernel[grid](
        x, y, out, N,
        eps=eps,
        BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.eps = eps
        # Pre-transpose weight to (K, N) for contiguous GEMM
        self.register_buffer("w_t", self.bmm.weight.detach().t().contiguous())

    def forward(self, x, y):
        x = x.contiguous()
        out = triton_linear(x, self.w_t, self.bmm.bias)
        return fused_norm_add_mul(out, y, self.eps)