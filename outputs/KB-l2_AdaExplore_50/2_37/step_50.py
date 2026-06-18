import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_swish_bias_kernel(
    A_ptr, B_ptr, Lbias_ptr, Bias_ptr, C_ptr,
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
        acc = tl.dot(a, b, acc, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Epilogue: + linear bias, swish, + extra bias
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    lbias = tl.load(Lbias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
    ebias = tl.load(Bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
    y = acc + lbias[None, :]
    y = y * tl.sigmoid(y) + ebias[None, :]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, y, mask=mask)


@triton.jit
def group_norm_kernel(
    Y_ptr, Gamma_ptr, Beta_ptr,
    M, N: tl.constexpr, C: tl.constexpr, G: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    col_base = pid_g * C
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    cols = col_base + offs_c

    y_ptrs = Y_ptr + pid_m * N + cols
    y = tl.load(y_ptrs, mask=mask_c, other=0.0)

    y_f = y.to(tl.float32)
    y_f_masked = tl.where(mask_c, y_f, 0.0)
    s = tl.sum(y_f_masked, axis=0)
    ss = tl.sum(y_f_masked * y_f_masked, axis=0)
    inv_c = 1.0 / C
    mean = s * inv_c
    var = ss * inv_c - mean * mean
    invstd = 1.0 / tl.sqrt(var + EPS)

    gamma = tl.load(Gamma_ptr + cols, mask=mask_c, other=0.0)
    beta = tl.load(Beta_ptr + cols, mask=mask_c, other=0.0)

    out = (y_f - mean) * invstd * gamma + beta
    tl.store(y_ptrs, out, mask=mask_c)


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
        x = x.contiguous().cuda()
        M, K = x.shape
        N = self.out_features
        G = self.num_groups
        C = N // G

        W = self.matmul.weight  # [N, K]
        Wt = W.t().contiguous()  # [K, N]
        Y = torch.empty((M, N), device=x.device, dtype=x.dtype)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
        gemm_swish_bias_kernel[grid](
            x, Wt, self.matmul.bias, self.bias, Y,
            M, N, K,
            x.stride(0), x.stride(1),
            Wt.stride(0), Wt.stride(1),
            Y.stride(0), Y.stride(1),
        )

        BLOCK_C = triton.next_power_of_2(C)
        if BLOCK_C < 16:
            BLOCK_C = 16

        gn_grid = (M, G)
        group_norm_kernel[gn_grid](
            Y, self.group_norm.weight, self.group_norm.bias,
            M, N, C, G,
            self.eps,
            BLOCK_C=BLOCK_C,
            num_warps=2,
            num_stages=2,
        )
        return Y