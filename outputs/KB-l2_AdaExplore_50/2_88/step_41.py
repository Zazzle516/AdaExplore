import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
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
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
    acc = acc + bias[None, :]
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_gemm_bias(x, w, bias):
    # x: (M, K), w: (N, K) -> out = x @ w.T + bias, shape (M, N)
    M, K = x.shape
    N, Kw = w.shape
    assert K == Kw
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # B = w.T: shape (K, N), stride_bk = w.stride(1)*N? Actually if w is contiguous (N, K),
    # then w.T has shape (K, N) with strides (1, K) i.e. stride_bk=1, stride_bn=K.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_bias_kernel[grid](
        x, w, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(1), w.stride(0),  # treat w as transposed: B[k,n] = w[n,k]
        out.stride(0), out.stride(1),
    )
    return out


@triton.jit
def fused_gn_swish_mul_swish_kernel(
    x_ptr, gamma_ptr, beta_ptr, mw_ptr, out_ptr,
    C: tl.constexpr, CH_PER_G: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_start = pid_g * CH_PER_G
    offs = tl.arange(0, BLOCK)
    mask = offs < CH_PER_G

    chan_off = group_start + offs
    row_base = pid_n * C

    x = tl.load(x_ptr + row_base + chan_off, mask=mask, other=0.0).to(tl.float32)

    sum_x = tl.sum(x, axis=0)
    mean = sum_x / CH_PER_G
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / CH_PER_G
    rstd = 1.0 / tl.sqrt(var + EPS)

    gamma = tl.load(gamma_ptr + chan_off, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + chan_off, mask=mask, other=0.0).to(tl.float32)
    mw = tl.load(mw_ptr + chan_off, mask=mask, other=0.0).to(tl.float32)

    y = xc * rstd * gamma + beta
    s1 = y * tl.sigmoid(y)
    z = s1 * mw
    out = z * tl.sigmoid(z)

    tl.store(out_ptr + row_base + chan_off, out, mask=mask)


def fused_gn_swish_mul_swish(x, gamma, beta, mw, num_groups, eps=1e-5):
    N, C = x.shape
    CH_PER_G = C // num_groups
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(CH_PER_G)
    if BLOCK < 32:
        BLOCK = 32
    grid = (N, num_groups)
    if BLOCK <= 64:
        nw = 1
    elif BLOCK <= 256:
        nw = 2
    else:
        nw = 4
    fused_gn_swish_mul_swish_kernel[grid](
        x, gamma, beta, mw, out,
        C, CH_PER_G,
        eps,
        BLOCK=BLOCK,
        num_warps=nw,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        self.num_groups = num_groups
        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous()
        # GEMM + bias via custom kernel
        y = triton_gemm_bias(x, self.gemm.weight, self.gemm.bias)
        out = fused_gn_swish_mul_swish(
            y,
            self.group_norm.weight,
            self.group_norm.bias,
            self.multiply_weight,
            self.num_groups,
            self.eps,
        )
        return out