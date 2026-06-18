import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_bias_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
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
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :].to(tl.float32)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


@triton.jit
def gn_swish_mul_swish_kernel(
    X_ptr,        # (M, N)
    W_ptr,        # (N,) gn weight
    B_ptr,        # (N,) gn bias
    MUL_ptr,      # (N,) multiply_weight
    Y_ptr,        # output (M, N)
    M, N, G,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_PROG: tl.constexpr,
    eps: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_gb = tl.program_id(1)

    g_start = pid_gb * GROUPS_PER_PROG
    # offsets [GROUPS_PER_PROG, GROUP_SIZE]
    off_g = tl.arange(0, GROUPS_PER_PROG)[:, None]
    off_c = tl.arange(0, GROUP_SIZE)[None, :]

    # absolute channel index within row: (g_start + off_g) * GROUP_SIZE + off_c
    chan_offs = (g_start + off_g) * GROUP_SIZE + off_c  # [GP, GS]
    row_off = pid_m * N
    ptrs = X_ptr + row_off + chan_offs

    x = tl.load(ptrs).to(tl.float32)  # [GP, GS]

    # mean/var along axis=1 per group row
    sum_x = tl.sum(x, axis=1)  # [GP]
    mean = sum_x / GROUP_SIZE
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / GROUP_SIZE
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W_ptr + chan_offs).to(tl.float32)
    b = tl.load(B_ptr + chan_offs).to(tl.float32)
    mul_w = tl.load(MUL_ptr + chan_offs).to(tl.float32)

    y = xc * rstd[:, None] * w + b
    y = y * tl.sigmoid(y)
    y = y * mul_w
    y = y * tl.sigmoid(y)

    tl.store(Y_ptr + row_off + chan_offs, y)


def triton_gemm_bias(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # B = weight.T, so stride_bk = weight.stride(1), stride_bn = weight.stride(0)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_bias_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        out.stride(0), out.stride(1),
    )
    return out


def triton_gn_swish_mul_swish(x, gn_w, gn_b, mul_w, num_groups, eps=1e-5):
    M, N = x.shape
    group_size = N // num_groups
    # Pack multiple groups per program
    if group_size <= 32:
        groups_per_prog = 8
    elif group_size <= 64:
        groups_per_prog = 4
    elif group_size <= 128:
        groups_per_prog = 2
    else:
        groups_per_prog = 1
    while num_groups % groups_per_prog != 0:
        groups_per_prog //= 2
    if groups_per_prog < 1:
        groups_per_prog = 1
    out = torch.empty_like(x)
    grid = (M, num_groups // groups_per_prog)
    total = group_size * groups_per_prog
    if total <= 64:
        nw = 1
    elif total <= 256:
        nw = 2
    elif total <= 1024:
        nw = 4
    else:
        nw = 8
    gn_swish_mul_swish_kernel[grid](
        x, gn_w, gn_b, mul_w, out,
        M, N, num_groups,
        GROUP_SIZE=group_size,
        GROUPS_PER_PROG=groups_per_prog,
        eps=eps,
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
        w = self.gemm.weight.contiguous()
        b = self.gemm.bias.contiguous()
        y = triton_gemm_bias(x, w, b)
        out = triton_gn_swish_mul_swish(
            y,
            self.group_norm.weight.contiguous(),
            self.group_norm.bias.contiguous(),
            self.multiply_weight.contiguous(),
            self.num_groups,
            self.eps,
        )
        return out