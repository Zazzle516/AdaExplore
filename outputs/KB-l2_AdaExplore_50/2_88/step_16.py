import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(
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
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        k_mask_a = k_offs[None, :] < K
        k_mask_b = k_offs[:, None] < K
        a = tl.load(a_ptrs, mask=m_mask & k_mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask_b & n_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_linear(x, weight_t, bias):
    # x: (M, K), weight_t: (K, N) (already transposed), bias: (N,)
    M, K = x.shape
    K2, N = weight_t.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_kernel[grid](
        x, weight_t, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


@triton.jit
def fused_gn_swish_mul_swish_kernel(
    x_ptr,          # (M, N)
    gamma_ptr,      # (N,)
    beta_ptr,       # (N,)
    mw_ptr,         # (N,)
    out_ptr,        # (M, N)
    M, N,
    G: tl.constexpr,            # num groups
    GROUP_SIZE: tl.constexpr,   # N // G
    GROUPS_PER_BLOCK: tl.constexpr,
    eps: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)  # which chunk of groups

    g_start = pid_g * GROUPS_PER_BLOCK
    # offsets within the row for this chunk of groups
    # shape: [GROUPS_PER_BLOCK, GROUP_SIZE]
    g_off = tl.arange(0, GROUPS_PER_BLOCK)  # group index within block
    c_off = tl.arange(0, GROUP_SIZE)        # channel within group

    # global channel index: (g_start + g_off) * GROUP_SIZE + c_off
    col_idx = (g_start + g_off)[:, None] * GROUP_SIZE + c_off[None, :]  # [GPB, GS]
    row_off = pid_m * N
    ptrs = x_ptr + row_off + col_idx

    g_mask = (g_start + g_off) < G  # [GPB]
    mask = g_mask[:, None]

    x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)

    # per-group mean/var: reduce along axis=1 (GROUP_SIZE)
    mean = tl.sum(x, axis=1) / GROUP_SIZE  # [GPB]
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / GROUP_SIZE  # [GPB]
    rstd = 1.0 / tl.sqrt(var + eps)  # [GPB]

    x_norm = xc * rstd[:, None]

    # load gamma, beta, mw for these channels
    gamma = tl.load(gamma_ptr + col_idx, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + col_idx, mask=mask, other=0.0).to(tl.float32)
    mw = tl.load(mw_ptr + col_idx, mask=mask, other=0.0).to(tl.float32)

    y = x_norm * gamma + beta
    # swish
    y = y * tl.sigmoid(y)
    # multiply
    y = y * mw
    # swish
    y = y * tl.sigmoid(y)

    tl.store(out_ptr + row_off + col_idx, y, mask=mask)


def fused_gn_swish_mul_swish(x, gamma, beta, mw, num_groups, eps=1e-5):
    M, N = x.shape
    GROUP_SIZE = N // num_groups
    G = num_groups
    out = torch.empty_like(x)

    # pick GROUPS_PER_BLOCK
    if GROUP_SIZE <= 16:
        GROUPS_PER_BLOCK = 64
    elif GROUP_SIZE <= 32:
        GROUPS_PER_BLOCK = 32
    elif GROUP_SIZE <= 64:
        GROUPS_PER_BLOCK = 8
    elif GROUP_SIZE <= 128:
        GROUPS_PER_BLOCK = 4
    else:
        GROUPS_PER_BLOCK = 1

    # ensure GROUPS_PER_BLOCK divides G or we mask
    num_g_blocks = (G + GROUPS_PER_BLOCK - 1) // GROUPS_PER_BLOCK

    grid = (M, num_g_blocks)

    num_warps = 4
    elements_per_block = GROUPS_PER_BLOCK * GROUP_SIZE
    if elements_per_block >= 512:
        num_warps = 8
    if elements_per_block <= 64:
        num_warps = 2

    fused_gn_swish_mul_swish_kernel[grid](
        x, gamma, beta, mw, out,
        M, N,
        G=G,
        GROUP_SIZE=GROUP_SIZE,
        GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
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
        # Pre-transpose weight for contiguous K-axis inner loads in GEMM
        with torch.no_grad():
            self.register_buffer('_weight_t', self.gemm.weight.t().contiguous())

    def forward(self, x):
        x = x.contiguous()
        weight_t = self._weight_t
        x = triton_linear(x, weight_t, self.gemm.bias)
        out = fused_gn_swish_mul_swish(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.multiply_weight,
            self.num_groups,
            self.eps,
        )
        return out