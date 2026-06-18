import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_swish_bias_kernel(
    x_ptr, w_ptr, b_ptr, bias2_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
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

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remain = K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=offs_k[None, :] < k_remain, other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < k_remain, other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # add linear bias
    b = tl.load(b_ptr + offs_n)
    acc = acc + b[None, :].to(tl.float32)

    # Swish: x * sigmoid(x)
    acc = acc * tl.sigmoid(acc)

    # add second bias
    b2 = tl.load(bias2_ptr + offs_n)
    acc = acc + b2[None, :].to(tl.float32)

    offs_m_o = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_o = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    o_ptrs = out_ptr + (offs_m_o[:, None] * stride_om + offs_n_o[None, :] * stride_on)
    tl.store(o_ptrs, acc, mask=(offs_m_o[:, None] < M) & (offs_n_o[None, :] < N))


@triton.jit
def group_norm_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    M, C, G,
    GS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_GS: tl.constexpr,
):
    # one program per (row, group)
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    offs = tl.arange(0, BLOCK_GS)
    mask = offs < GS

    base = pid_m * C + pid_g * GS
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    sum_x = tl.sum(x, axis=0)
    mean = sum_x / GS
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / GS
    rstd = 1.0 / tl.sqrt(var + EPS)

    gamma = tl.load(gamma_ptr + pid_g * GS + offs, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + pid_g * GS + offs, mask=mask, other=0.0)

    y = xc * rstd * gamma + beta
    tl.store(out_ptr + base + offs, y, mask=mask)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous()
        M, K = x.shape
        N = self.out_features
        W = self.matmul.weight  # (N, K)
        b = self.matmul.bias    # (N,)
        bias2 = self.bias       # (N,)

        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
        gemm_swish_bias_kernel[grid](
            x, W, b, bias2, out,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            out.stride(0), out.stride(1),
        )

        # Group norm
        G = self.num_groups
        GS = N // G
        BLOCK_GS = _next_pow2(GS)
        out2 = torch.empty_like(out)
        gn_grid = (M, G)
        group_norm_kernel[gn_grid](
            out, self.group_norm.weight, self.group_norm.bias, out2,
            M, N, G,
            GS=GS,
            EPS=self.eps,
            BLOCK_GS=BLOCK_GS,
        )
        return out2