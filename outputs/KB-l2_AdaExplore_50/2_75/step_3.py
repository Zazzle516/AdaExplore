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
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(
    A, B, C, bias,
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

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias_vals = tl.load(bias + offs_n, mask=mask_n, other=0.0)
    acc += bias_vals[None, :]

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def gn_min_bias_kernel(
    X_ptr, gamma_ptr, beta_ptr, bias_ptr, OUT_ptr,
    M, N, num_groups, eps,
    BLOCK_C: tl.constexpr,
):
    # one program per (row, group)
    row = tl.program_id(0)
    g = tl.program_id(1)

    C_per_g = N // num_groups
    # offsets within group
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C_per_g

    base = row * N + g * C_per_g
    x = tl.load(X_ptr + base + offs, mask=mask, other=0.0)

    # compute mean and var
    sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
    mean = sum_x / C_per_g
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / C_per_g
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(gamma_ptr + g * C_per_g + offs, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + g * C_per_g + offs, mask=mask, other=0.0)

    y = (x - mean) * rstd * gamma + beta
    # take min over the group, then we need min over whole row -> atomic min via second pass
    # store normalized value temporarily
    # but we need min over full row N. So we need different approach.
    # We'll compute min over this group and atomic min into a scratch.
    y_masked = tl.where(mask, y, float('inf'))
    group_min = tl.min(y_masked, axis=0)

    # atomic min into OUT_ptr[row]  (will be combined later)
    # OUT_ptr is shape (M,) scratch for min
    tl.atomic_min(OUT_ptr + row, group_min)


@triton.jit
def add_bias_kernel(
    min_ptr, bias_ptr, out_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    m = tl.load(min_ptr + row)
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
    out = m + b
    tl.store(out_ptr + row * N + offs, out, mask=mask)


def triton_gemm(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    # weight is (N, K), we treat B as (K, N) by using stride accordingly
    gemm_kernel[grid](
        x, weight, out, bias,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # B = W^T effectively
        out.stride(0), out.stride(1),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups

    def forward(self, x):
        x = x.contiguous()
        # GEMM + bias
        y = triton_gemm(x, self.gemm.weight, self.gemm.bias)

        M, N = y.shape
        C_per_g = N // self.num_groups
        # next power of 2 for BLOCK_C
        BLOCK_C = 1
        while BLOCK_C < C_per_g:
            BLOCK_C *= 2

        # min scratch initialized to +inf
        min_buf = torch.full((M,), float('inf'), device=y.device, dtype=y.dtype)

        gn_min_bias_kernel[(M, self.num_groups)](
            y, self.group_norm.weight, self.group_norm.bias, None, min_buf,
            M, N, self.num_groups, self.group_norm.eps,
            BLOCK_C=BLOCK_C,
        )

        # bias is (1, out_features, 1, 1) -> output should be (M, out_features, 1, 1)?
        # Original: x has shape (M, 1) after min keepdim, then x + bias broadcasts.
        # x shape (M, 1), bias shape (1, out_features, 1, 1). Broadcast -> (1, out_features, M, 1)? 
        # Let's check: (M,1) + (1, of, 1, 1) - broadcasting from right: 
        # (M,1) -> (1,1,M,1)? No, broadcasting aligns from right.
        # (M, 1) has 2 dims, (1, of, 1, 1) has 4 dims. Pad (M,1) -> (1,1,M,1).
        # Result: (1, of, M, 1).
        bias_flat = self.bias.view(-1)  # (out_features,)
        out_features = bias_flat.shape[0]
        # output shape (1, of, M, 1)
        out = torch.empty((1, out_features, M, 1), device=y.device, dtype=y.dtype)
        # out[0, c, m, 0] = min_buf[m] + bias_flat[c]
        # we can use broadcasting through torch directly since this is just a small elementwise
        out = min_buf.view(1, 1, M, 1) + self.bias  # bias is (1, of, 1, 1) -> result (1, of, M, 1)
        return out