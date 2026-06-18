import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr, bias_ptr,
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

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_gemm(x, weight, bias):
    # x: [M, K], weight: [N, K], bias: [N]
    M, K = x.shape
    N, _ = weight.shape
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # B = weight.T, so stride_bk = 1, stride_bn = K
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_kernel[grid](
        x, weight, out, bias,
        M, N, K,
        x.stride(0), x.stride(1),
        1, weight.stride(0),
        out.stride(0), out.stride(1),
    )
    return out


@triton.jit
def gn_min_kernel(
    X_ptr, W_ptr, B_ptr, Min_ptr,
    M, N,
    eps,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # one program per row
    row = tl.program_id(0)

    offs = tl.arange(0, BLOCK_N)
    x = tl.load(X_ptr + row * N + offs).to(tl.float32)
    w = tl.load(W_ptr + offs).to(tl.float32)
    b = tl.load(B_ptr + offs).to(tl.float32)

    x_2d = tl.reshape(x, (NUM_GROUPS, GROUP_SIZE))
    mean = tl.sum(x_2d, axis=1) / GROUP_SIZE
    diff = x_2d - mean[:, None]
    var = tl.sum(diff * diff, axis=1) / GROUP_SIZE
    rstd = 1.0 / tl.sqrt(var + eps)

    x_norm = diff * rstd[:, None]
    x_norm_flat = tl.reshape(x_norm, (BLOCK_N,))

    y = x_norm_flat * w + b
    min_val = tl.min(y, axis=0)

    tl.store(Min_ptr + row, min_val)


@triton.jit
def bias_add_kernel(
    Min_ptr, Bias_ptr, Out_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    # output shape [N, M], one program per row of N
    row = tl.program_id(0)  # m index
    n_off = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = n_off < N
    m_val = tl.load(Min_ptr + row)
    b = tl.load(Bias_ptr + n_off, mask=mask, other=0.0)
    out = m_val + b
    # output layout: [1, N, M, 1] -> stored as N*M with stride [M, 1]
    # out[n, m] = m_val[m] + b[n]
    tl.store(Out_ptr + n_off * M + row, out, mask=mask)


def triton_gn_min(x, weight, bias, num_groups, eps):
    M, N = x.shape
    GROUP_SIZE = N // num_groups
    out = torch.empty((M,), device=x.device, dtype=x.dtype)
    grid = (M,)
    gn_min_kernel[grid](
        x, weight, bias, out,
        M, N, eps,
        NUM_GROUPS=num_groups,
        GROUP_SIZE=GROUP_SIZE,
        BLOCK_N=N,
        num_warps=8,
        num_stages=2,
    )
    return out


def triton_fused_bias_add(min_vals, bias_flat, M, N):
    # output: [N, M] which views as [1, N, M, 1]
    out = torch.empty((N, M), device=min_vals.device, dtype=min_vals.dtype)
    BLOCK_N = 256
    grid = (M, triton.cdiv(N, BLOCK_N))
    bias_add_kernel[grid](
        min_vals, bias_flat, out,
        M, N,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups

        # GEMM params
        gemm = nn.Linear(in_features, out_features)
        self.gemm_weight = nn.Parameter(gemm.weight.detach().clone())
        self.gemm_bias = nn.Parameter(gemm.bias.detach().clone())

        # GroupNorm params
        gn = nn.GroupNorm(num_groups, out_features)
        self.gn_weight = nn.Parameter(gn.weight.detach().clone())
        self.gn_bias = nn.Parameter(gn.bias.detach().clone())
        self.eps = 1e-5

        # Final bias
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.contiguous()
        M = x.shape[0]
        N = self.out_features
        # GEMM: [M, K] x [K, N] + bias -> [M, N]
        y = triton_gemm(x, self.gemm_weight, self.gemm_bias)
        # GN + min over dim=1 keepdim -> [M]
        m = triton_gn_min(y, self.gn_weight, self.gn_bias, self.num_groups, self.eps)
        # Fused bias broadcast add: produce [N, M] then view as [1, N, M, 1]
        bias_flat = self.bias.view(-1)
        out_flat = triton_fused_bias_add(m, bias_flat, M, N)
        return out_flat.view(1, N, M, 1)