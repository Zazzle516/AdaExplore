import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_swish_bias_kernel(
    A_ptr, B_ptr, bias_lin_ptr, bias_extra_ptr, C_ptr,
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
        a_mask = mask_m[:, None] & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & mask_n[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias_lin = tl.load(bias_lin_ptr + offs_n, mask=mask_n, other=0.0)
    bias_extra = tl.load(bias_extra_ptr + offs_n, mask=mask_n, other=0.0)

    x = acc + bias_lin[None, :]
    # Swish: x * sigmoid(x)
    swish = x * tl.sigmoid(x)
    out = swish + bias_extra[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, out, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['CG'],
)
@triton.jit
def group_norm_kernel(
    X_ptr, Y_ptr, weight_ptr, bias_ptr,
    M, C, G, CG,
    eps,
    BLOCK: tl.constexpr,
):
    # one program per (row, group)
    pid = tl.program_id(0)
    row = pid // G
    group = pid % G

    base = row * C + group * CG
    offs = tl.arange(0, BLOCK)
    mask = offs < CG

    x = tl.load(X_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    sum_x = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)
    inv_cg = 1.0 / CG.to(tl.float32)
    mean = sum_x * inv_cg
    var = sum_x2 * inv_cg - mean * mean
    rstd = tl.rsqrt(var + eps)

    w = tl.load(weight_ptr + group * CG + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + group * CG + offs, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * w + b
    tl.store(Y_ptr + base + offs, y, mask=mask)


def next_pow2(n):
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

        lin = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(lin.weight.data.clone())
        self.lin_bias = nn.Parameter(lin.bias.data.clone())
        self.bias = nn.Parameter(torch.randn(bias_shape))

        gn = nn.GroupNorm(num_groups, out_features)
        self.gn_weight = nn.Parameter(gn.weight.data.clone())
        self.gn_bias = nn.Parameter(gn.bias.data.clone())
        self.eps = 1e-5

        self.CG = out_features // num_groups
        self.BLOCK_GN = next_pow2(self.CG)

    def forward(self, x):
        x = x.contiguous()
        M, K = x.shape
        N = self.out_features
        # B = weight.T -> (K, N)
        W = self.weight  # (N, K)
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
        gemm_swish_bias_kernel[grid](
            x, W, self.lin_bias, self.bias, out,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(1), W.stride(0),  # B = W.T: stride_bk = W.stride(1) (over K), stride_bn = W.stride(0) (over N)
            out.stride(0), out.stride(1),
        )

        y = torch.empty_like(out)
        grid2 = (M * self.num_groups,)
        group_norm_kernel[grid2](
            out, y, self.gn_weight, self.gn_bias,
            M, N, self.num_groups, self.CG,
            self.eps,
            BLOCK=self.BLOCK_GN,
        )
        return y