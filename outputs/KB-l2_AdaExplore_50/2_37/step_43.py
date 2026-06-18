import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


MATMUL_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=MATMUL_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def matmul_swish_bias_kernel(
    A_ptr, B_ptr, C_ptr, bias_lin_ptr, bias_extra_ptr,
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
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias_lin = tl.load(bias_lin_ptr + offs_n)
    acc += bias_lin[None, :]

    sig = 1.0 / (1.0 + tl.exp(-acc))
    acc = acc * sig

    bias_extra = tl.load(bias_extra_ptr + offs_n)
    acc += bias_extra[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc)


@triton.jit
def group_norm_row_kernel(
    X_ptr, Y_ptr, weight_ptr, bias_ptr,
    C, num_groups, eps,
    ROWS_PER_PROG: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG

    g_offs = tl.arange(0, NUM_GROUPS)[:, None]
    e_offs = tl.arange(0, GROUP_SIZE)[None, :]
    col_offs = g_offs * GROUP_SIZE + e_offs  # [NUM_GROUPS, GROUP_SIZE]

    w = tl.load(weight_ptr + col_offs)
    b = tl.load(bias_ptr + col_offs)

    inv_gs = 1.0 / GROUP_SIZE.to(tl.float32)

    for i in tl.static_range(0, ROWS_PER_PROG):
        row = row_start + i
        base = row * C
        x = tl.load(X_ptr + base + col_offs).to(tl.float32)
        # per-group mean
        sum_x = tl.sum(x, axis=1, keep_dims=True)
        mean = sum_x * inv_gs
        xc = x - mean
        sum_xx = tl.sum(xc * xc, axis=1, keep_dims=True)
        var = sum_xx * inv_gs
        rstd = 1.0 / tl.sqrt(var + eps)
        y = xc * rstd * w + b
        tl.store(Y_ptr + base + col_offs, y)


def matmul_swish_bias(x, weight, bias_lin, bias_extra):
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    stride_am, stride_ak = x.stride()
    stride_bk = 1
    stride_bn = weight.stride(0)
    stride_cm, stride_cn = out.stride()

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    matmul_swish_bias_kernel[grid](
        x, weight, out, bias_lin, bias_extra,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        GROUP_M=8,
    )
    return out


def group_norm_fwd(x, weight, bias, num_groups, eps):
    M, C = x.shape
    group_size = C // num_groups
    out = torch.empty_like(x)

    ROWS_PER_PROG = 4
    assert M % ROWS_PER_PROG == 0
    grid = (M // ROWS_PER_PROG,)
    group_norm_row_kernel[grid](
        x, out, weight, bias,
        C, num_groups, eps,
        ROWS_PER_PROG=ROWS_PER_PROG,
        NUM_GROUPS=num_groups,
        GROUP_SIZE=group_size,
        num_warps=8,
        num_stages=3,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.num_groups = num_groups
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous()
        w = self.matmul.weight.contiguous()
        b_lin = self.matmul.bias.contiguous()
        b_extra = self.bias.contiguous()
        y = matmul_swish_bias(x, w, b_lin, b_extra)
        y = group_norm_fwd(y, self.group_norm.weight, self.group_norm.bias,
                           self.num_groups, self.group_norm.eps)
        return y