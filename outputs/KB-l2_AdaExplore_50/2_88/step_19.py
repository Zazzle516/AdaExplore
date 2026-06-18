import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
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
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
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
    X_ptr, W_ptr, B_ptr, MUL_ptr, Y_ptr,
    M, N,
    GROUP_SIZE: tl.constexpr,
    G_PER_PROG: tl.constexpr,
    eps: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g_block = tl.program_id(1)

    # Each program processes G_PER_PROG groups for one row
    # tile shape: (G_PER_PROG, GROUP_SIZE)
    g_offs = tl.arange(0, G_PER_PROG)
    c_offs = tl.arange(0, GROUP_SIZE)

    g_start = pid_g_block * G_PER_PROG
    # column offsets in the row for our tile: (G_PER_PROG, GROUP_SIZE)
    n_offs = (g_start + g_offs)[:, None] * GROUP_SIZE + c_offs[None, :]

    row_base = pid_m * N
    ptrs = X_ptr + row_base + n_offs

    x = tl.load(ptrs).to(tl.float32)

    # mean/var per group (reduce along axis=1)
    mean = tl.sum(x, axis=1) / GROUP_SIZE  # (G_PER_PROG,)
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / GROUP_SIZE
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W_ptr + n_offs).to(tl.float32)
    b = tl.load(B_ptr + n_offs).to(tl.float32)
    mul_w = tl.load(MUL_ptr + n_offs).to(tl.float32)

    y = xc * rstd[:, None] * w + b
    y = y * tl.sigmoid(y)
    y = y * mul_w
    y = y * tl.sigmoid(y)

    tl.store(Y_ptr + row_base + n_offs, y)


def triton_gemm_bias(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
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
    # choose G_PER_PROG so tile has reasonable size; group_size=32, choose 8 -> 256 elements
    if group_size <= 32:
        G_PER_PROG = 8
        num_warps = 4
    elif group_size <= 64:
        G_PER_PROG = 4
        num_warps = 4
    else:
        G_PER_PROG = 2
        num_warps = 4

    while num_groups % G_PER_PROG != 0:
        G_PER_PROG //= 2
    if G_PER_PROG < 1:
        G_PER_PROG = 1

    out = torch.empty_like(x)
    grid = (M, num_groups // G_PER_PROG)
    gn_swish_mul_swish_kernel[grid](
        x, gn_w, gn_b, mul_w, out,
        M, N,
        GROUP_SIZE=group_size,
        G_PER_PROG=G_PER_PROG,
        eps=eps,
        num_warps=num_warps,
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