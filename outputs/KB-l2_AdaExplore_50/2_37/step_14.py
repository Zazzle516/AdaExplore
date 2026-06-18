import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_swish_bias_kernel(
    A_ptr, B_ptr, C_ptr, Bias_ptr, Bias2_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

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

    # add linear bias
    bias = tl.load(Bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]
    # swish
    acc = acc * tl.sigmoid(acc)
    # add second bias
    bias2 = tl.load(Bias2_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias2[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def group_norm_kernel(
    X_ptr, Y_ptr, Gamma_ptr, Beta_ptr,
    M, C, G, GS,
    eps,
    BLOCK_GS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    # one program per (row, group)
    row = tl.program_id(0)
    g = tl.program_id(1)

    offs = tl.arange(0, BLOCK_GS)
    mask = offs < GS

    x_ptrs = X_ptr + row * C + g * GS + offs
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / GS
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / GS
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(Gamma_ptr + g * GS + offs, mask=mask, other=0.0)
    beta = tl.load(Beta_ptr + g * GS + offs, mask=mask, other=0.0)

    y = (x - mean) * rstd * gamma + beta
    tl.store(Y_ptr + row * C + g * GS + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.group_size = out_features // num_groups

        # Linear params
        lin = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(lin.weight.data.clone())
        self.linear_bias = nn.Parameter(lin.bias.data.clone())

        # extra bias
        self.bias = nn.Parameter(torch.randn(bias_shape))

        # GroupNorm params
        gn = nn.GroupNorm(num_groups, out_features)
        self.gn_weight = nn.Parameter(gn.weight.data.clone())
        self.gn_bias = nn.Parameter(gn.bias.data.clone())
        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous().cuda()
        M, K = x.shape
        N = self.out_features

        # B = weight.T  (N x K -> K x N)
        W = self.weight  # (N, K)
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
        gemm_swish_bias_kernel[grid](
            x, W, out, self.linear_bias, self.bias,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(1), W.stride(0),  # B = W.T -> stride_bk=W.stride(1), stride_bn=W.stride(0)
            out.stride(0), out.stride(1),
        )

        # GroupNorm
        y = torch.empty_like(out)
        GS = self.group_size
        # pick BLOCK_GS as next pow2 >= GS
        BLOCK_GS = 1
        while BLOCK_GS < GS:
            BLOCK_GS *= 2

        grid2 = (M, self.num_groups)
        group_norm_kernel[grid2](
            out, y, self.gn_weight, self.gn_bias,
            M, N, self.num_groups, GS,
            self.eps,
            BLOCK_GS=BLOCK_GS,
            NUM_GROUPS=self.num_groups,
            num_warps=2,
        )
        return y