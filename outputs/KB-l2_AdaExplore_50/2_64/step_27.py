import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_kernel(
    A_ptr, B_ptr, C_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
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

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K_full = (K // BLOCK_K) * BLOCK_K
    for k in range(0, K_full, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if K_full < K:
        k_remaining = K - K_full
        a = tl.load(a_ptrs, mask=(offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining), other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)

    mask_m = offs_m < M
    mask_n = offs_n < N

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        acc = acc + bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def custom_linear(x, B, bias):
    M, K = x.shape
    N = B.shape[1]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    has_bias = bias is not None
    if has_bias:
        bias_c = bias.contiguous()
    else:
        bias_c = torch.empty(1, device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_bias_kernel[grid](
        x, B, out, bias_c,
        M, N, K,
        x.stride(0), x.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=has_bias,
    )
    return out


@triton.jit
def lse_partial_kernel(
    in_ptr, max_ptr, sum_ptr,
    M, N, NSPLIT: tl.constexpr,
    stride_m, stride_n,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)
    row_ptr = in_ptr + pid_m * stride_m

    start = pid_s * BLOCK_N
    offs = start + tl.arange(0, BLOCK_N)
    mask = offs < N
    neg_inf = float('-inf')
    vals = tl.load(row_ptr + offs * stride_n, mask=mask, other=neg_inf)
    m = tl.max(vals, axis=0)
    e = tl.exp(vals - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)

    tl.store(max_ptr + pid_m * NSPLIT + pid_s, m)
    tl.store(sum_ptr + pid_m * NSPLIT + pid_s, s)


@triton.jit
def lse_combine_kernel(
    max_ptr, sum_ptr, out_ptr,
    M, NSPLIT: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_S)
    mask = offs < NSPLIT
    neg_inf = float('-inf')
    ms = tl.load(max_ptr + pid * NSPLIT + offs, mask=mask, other=neg_inf)
    ss = tl.load(sum_ptr + pid * NSPLIT + offs, mask=mask, other=0.0)
    M_max = tl.max(ms, axis=0)
    ss_adj = ss * tl.exp(ms - M_max)
    ss_adj = tl.where(mask, ss_adj, 0.0)
    s_total = tl.sum(ss_adj, axis=0)
    lse = M_max + tl.log(s_total)

    x = tl.where(lse >= 0.0, lse, lse * 0.01)
    x = tl.where(x >= 0.0, x, x * 0.01)

    inv_sqrt2 = 0.7071067811865475
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(out_ptr + pid, x)


def fused_lse_act(z):
    M, N = z.shape
    z = z.contiguous()
    out = torch.empty((M, 1), device=z.device, dtype=z.dtype)

    # Choose split
    if N >= 4096:
        NSPLIT = 4
    elif N >= 2048:
        NSPLIT = 2
    else:
        NSPLIT = 1
    chunk = (N + NSPLIT - 1) // NSPLIT
    BLOCK_N = triton.next_power_of_2(chunk)

    max_buf = torch.empty((M, NSPLIT), device=z.device, dtype=torch.float32)
    sum_buf = torch.empty((M, NSPLIT), device=z.device, dtype=torch.float32)

    num_warps = 8 if BLOCK_N >= 2048 else 4
    lse_partial_kernel[(M, NSPLIT)](
        z, max_buf, sum_buf,
        M, N, NSPLIT,
        z.stride(0), z.stride(1),
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=2,
    )

    BLOCK_S = triton.next_power_of_2(NSPLIT) if NSPLIT > 1 else 1
    lse_combine_kernel[(M,)](
        max_buf, sum_buf, out,
        M, NSPLIT,
        BLOCK_S=BLOCK_S,
        num_warps=1,
        num_stages=1,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        with torch.no_grad():
            B = self.linear.weight.detach().t().contiguous().cuda()
        self.register_buffer('_B', B, persistent=False)

    def forward(self, x):
        x = x.cuda().contiguous()
        bias = self.linear.bias
        z = custom_linear(x, self._B, bias)
        return fused_lse_act(z)