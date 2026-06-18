import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
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
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
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
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N,
    eps,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # one program per row
    row = tl.program_id(0)
    if row >= M:
        return

    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)

    # Compute per-group mean / var
    # Reshape conceptually: x is [NUM_GROUPS, GROUP_SIZE]
    # Build group ids
    group_id = offs // GROUP_SIZE  # [BLOCK_N]

    # sum per group via masking — use a loop over groups since NUM_GROUPS is constexpr
    # but NUM_GROUPS = 512 is large; better: compute mean/var with segmented approach using tl.reshape
    x_2d = tl.reshape(x, (NUM_GROUPS, GROUP_SIZE))
    mean = tl.sum(x_2d, axis=1) / GROUP_SIZE  # [NUM_GROUPS]
    diff = x_2d - mean[:, None]
    var = tl.sum(diff * diff, axis=1) / GROUP_SIZE
    rstd = 1.0 / tl.sqrt(var + eps)

    # normalize
    x_norm = diff * rstd[:, None]
    x_norm_flat = tl.reshape(x_norm, (BLOCK_N,))

    # apply affine
    y = x_norm_flat * w + b

    # min reduction over N
    y_masked = tl.where(mask, y, float('inf'))
    min_val = tl.min(y_masked, axis=0)

    tl.store(Out_ptr + row, min_val)


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
        # GEMM: [M, K] x [K, N] + bias -> [M, N]
        y = triton_gemm(x, self.gemm_weight, self.gemm_bias)
        # GN + min over dim=1 keepdim -> [M, 1]
        m = triton_gn_min(y, self.gn_weight, self.gn_bias, self.num_groups, self.eps)
        # reshape min to [M, 1] then to [1, 1, M, 1] for broadcast with [1, N, 1, 1]
        m = m.view(1, 1, -1, 1)
        # bias: [1, N, 1, 1] (originally (1, out_features, 1, 1))
        out = m + self.bias
        return out