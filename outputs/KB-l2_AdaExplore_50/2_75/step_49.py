import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gn_min_kernel(
    X_ptr, Gamma_ptr, Beta_ptr, Out_ptr,
    M, N,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    eps: tl.constexpr,
):
    row = tl.program_id(0)

    g_off = tl.arange(0, NUM_GROUPS)
    c_off = tl.arange(0, GROUP_SIZE)
    offs = g_off[:, None] * GROUP_SIZE + c_off[None, :]
    ptrs = X_ptr + row * N + offs

    x = tl.load(ptrs).to(tl.float32)
    gamma = tl.load(Gamma_ptr + offs).to(tl.float32)
    beta = tl.load(Beta_ptr + offs).to(tl.float32)

    s = tl.sum(x, axis=1)
    mean = s / GROUP_SIZE
    xc = x - mean[:, None]
    sq = tl.sum(xc * xc, axis=1)
    var = sq / GROUP_SIZE
    inv = 1.0 / tl.sqrt(var + eps)

    norm = xc * inv[:, None] * gamma + beta

    m1 = tl.min(norm, axis=1)
    res = tl.min(m1, axis=0)
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

    bias = tl.load(Bias_ptr + offs_n, mask=mask_n, other=0.0)
    minv = tl.load(MinVal_ptr + offs_m, mask=mask_m, other=0.0)

    out = bias[:, None] + minv[None, :]

    out_ptrs = Out_ptr + offs_n[:, None] * M + offs_m[None, :]
    mask = mask_n[:, None] & mask_m[None, :]
    tl.store(out_ptrs, out, mask=mask)


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

    def forward(self, x):
        M = x.shape[0]
        N = self.out_features

        # Use cuBLAS for GEMM (hard to beat on fp32 8192x8192)
        gemm_out = F.linear(x, self.gemm.weight, self.gemm.bias)

        min_vals = torch.empty((M,), device=x.device, dtype=x.dtype)

        gn_min_kernel[(M,)](
            gemm_out, self.group_norm.weight, self.group_norm.bias, min_vals,
            M, N,
            self.num_groups, self.group_size,
            self.eps,
            num_warps=4,
        )

        bias_flat = self.bias.view(-1).contiguous()
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