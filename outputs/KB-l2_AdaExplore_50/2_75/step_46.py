import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_bias_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
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

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

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

    bias = tl.load(Bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


# Fused GroupNorm + min along channel dim (per row produces a scalar)
# Layout: each row has N = NUM_GROUPS * GROUP_SIZE elements
# Reshape to [NUM_GROUPS, GROUP_SIZE], compute per-group mean/var,
# normalize, multiply by gamma + beta, then take row-wide min.
@triton.jit
def gn_min_kernel(
    X_ptr, Gamma_ptr, Beta_ptr, Out_ptr,
    M, N,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    eps,
):
    row = tl.program_id(0)

    # Build 2D index: [NUM_GROUPS, GROUP_SIZE]
    g_off = tl.arange(0, NUM_GROUPS)  # [G]
    c_off = tl.arange(0, GROUP_SIZE)  # [C]
    # Combined offsets: [G, C]
    offs = g_off[:, None] * GROUP_SIZE + c_off[None, :]
    ptrs = X_ptr + row * N + offs

    x = tl.load(ptrs).to(tl.float32)  # [G, C]
    gamma = tl.load(Gamma_ptr + offs).to(tl.float32)
    beta = tl.load(Beta_ptr + offs).to(tl.float32)

    # Per-group mean/var, reduce along axis=1 (channel within group)
    s = tl.sum(x, axis=1)  # [G]
    mean = s / GROUP_SIZE  # [G]
    xc = x - mean[:, None]
    sq = tl.sum(xc * xc, axis=1)  # [G]
    var = sq / GROUP_SIZE
    inv = 1.0 / tl.sqrt(var + eps)  # [G]

    norm = xc * inv[:, None] * gamma + beta  # [G, C]

    # Row-wide min
    m1 = tl.min(norm, axis=1)  # [G]
    res = tl.min(m1, axis=0)  # scalar
    tl.store(Out_ptr + row, res)


@triton.jit
def add_bias_kernel(
    MinVal_ptr, Bias_ptr, Out_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    mask_n = offs_n < N
    mask_m = offs_m < M

    bias = tl.load(Bias_ptr + offs_n, mask=mask_n, other=0.0)  # [BN]
    minv = tl.load(MinVal_ptr + offs_m, mask=mask_m, other=0.0)  # [BM]

    out = bias[:, None] + minv[None, :]  # [BN, BM]

    # output layout [1, N, M, 1] -> flat index n * M + m
    out_ptrs = Out_ptr + offs_n[:, None] * M + offs_m[None, :]
    mask = mask_n[:, None] & mask_m[None, :]
    tl.store(out_ptrs, out, mask=mask)


def triton_gemm_bias(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    x = x.contiguous()
    w_t = weight.t().contiguous()
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    gemm_bias_kernel[grid](
        x, w_t, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.group_size = out_features // num_groups
        self.eps = 1e-5

        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

        # Pre-flatten the weight transpose for GEMM and bias for add
        self._w_t = None

    def forward(self, x):
        M = x.shape[0]
        N = self.out_features

        gemm_out = triton_gemm_bias(x.contiguous(), self.gemm.weight, self.gemm.bias)

        min_vals = torch.empty((M,), device=x.device, dtype=x.dtype)

        gn_min_kernel[(M,)](
            gemm_out, self.group_norm.weight, self.group_norm.bias, min_vals,
            M, N,
            self.num_groups, self.group_size,
            self.eps,
            num_warps=4,
        )

        bias_flat = self.bias.view(-1).contiguous()  # [N]
        out = torch.empty((1, N, M, 1), device=x.device, dtype=x.dtype)

        BLOCK_M = 64
        BLOCK_N = 64
        grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M))
        add_bias_kernel[grid](
            min_vals, bias_flat, out,
            M, N,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return out