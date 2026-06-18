import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gn_min_kernel(
    y_ptr, gn_w_ptr, gn_b_ptr, min_out_ptr,
    M, eps,
    N: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    offs_n = tl.arange(0, N)
    y = tl.load(y_ptr + row * N + offs_n).to(tl.float32)

    y_2d = tl.reshape(y, (NUM_GROUPS, GROUP_SIZE))
    mean = tl.sum(y_2d, axis=1) / GROUP_SIZE
    diff = y_2d - mean[:, None]
    var = tl.sum(diff * diff, axis=1) / GROUP_SIZE
    rstd = 1.0 / tl.sqrt(var + eps)
    y_norm_2d = diff * rstd[:, None]
    y_norm = tl.reshape(y_norm_2d, (N,))

    gn_w = tl.load(gn_w_ptr + offs_n)
    gn_b = tl.load(gn_b_ptr + offs_n)
    y_norm = y_norm * gn_w + gn_b

    min_val = tl.min(y_norm, axis=0)
    tl.store(min_out_ptr + row, min_val)


@triton.jit
def add_bias_kernel(
    min_ptr, bias_ptr, out_ptr,
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

    bias_val = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    min_vals = tl.load(min_ptr + offs_m, mask=mask_m, other=0.0)

    out_vals = min_vals[None, :] + bias_val[:, None]

    out_ptrs = out_ptr + offs_n[:, None] * M + offs_m[None, :]
    mask = mask_n[:, None] & mask_m[None, :]
    tl.store(out_ptrs, out_vals, mask=mask)


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
        x = x.cuda().contiguous()
        M, K = x.shape
        N = self.out_features

        y = F.linear(x, self.gemm.weight, self.gemm.bias)
        y = y.contiguous()

        gn_w = self.group_norm.weight.contiguous()
        gn_b = self.group_norm.bias.contiguous()
        bias = self.bias.contiguous().view(-1)

        min_out = torch.empty((M,), device=x.device, dtype=x.dtype)
        grid_gn = (M,)
        gn_min_kernel[grid_gn](
            y, gn_w, gn_b, min_out,
            M, self.eps,
            N=N,
            NUM_GROUPS=self.num_groups,
            GROUP_SIZE=self.group_size,
            num_warps=8,
            num_stages=3,
        )

        out_flat = torch.empty((N, M), device=x.device, dtype=x.dtype)
        BLOCK_M = 128
        BLOCK_N = 128
        grid_b = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M))
        add_bias_kernel[grid_b](
            min_out, bias, out_flat,
            M, N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=3,
        )

        return out_flat.view(1, N, M, 1)