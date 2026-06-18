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

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        mask_k = (k + offs_k) < K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def gn_min_kernel(
    X_ptr, W_ptr, B_ptr, OUT_ptr,
    M, N,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x_ptrs = X_ptr + pid * N + offs
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # Compute per-group mean / var using grouped reduction
    # Reshape into (NUM_GROUPS, GROUP_SIZE) by treating index = g*GROUP_SIZE + i
    # We'll do this via a 2D arange.
    # But we need to load groups separately. Loop over groups in a single program.

    row_min = tl.full((1,), float('inf'), dtype=tl.float32)

    # Use a loop over groups
    inv_gs = 1.0 / GROUP_SIZE

    # We will compute everything by iterating group by group
    # Load weight and bias for full row
    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)

    # For each element, find its group id
    group_ids = offs // GROUP_SIZE  # shape [BLOCK_N]

    # Compute per-group sum and sum-of-squares using segmented reduction
    # We'll loop over groups
    out_row_min = float('inf')

    for g in range(0, NUM_GROUPS):
        group_mask = (group_ids == g) & mask
        x_g = tl.where(group_mask, x, 0.0)
        x_g_sq = tl.where(group_mask, x * x, 0.0)
        sum_x = tl.sum(x_g, axis=0)
        sum_x2 = tl.sum(x_g_sq, axis=0)
        mean = sum_x * inv_gs
        var = sum_x2 * inv_gs - mean * mean
        rstd = 1.0 / tl.sqrt(var + eps)
        # normalized for this group
        norm = (x - mean) * rstd
        y = norm * w + b
        # take min only over this group's elements
        y_for_min = tl.where(group_mask, y, float('inf'))
        g_min = tl.min(y_for_min, axis=0)
        out_row_min = tl.minimum(out_row_min, g_min)

    tl.store(OUT_ptr + pid, out_row_min)


def triton_gemm(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # B = weight.T, so stride_bk = 1, stride_bn = K
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    gemm_kernel[grid](
        x, weight, out, bias,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # B = W^T
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

        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous()
        M = x.shape[0]
        N = self.out_features

        # GEMM + bias fused
        gemm_out = triton_gemm(x, self.gemm.weight, self.gemm.bias)

        # GN + per-row min fused
        row_min = torch.empty((M,), device=x.device, dtype=x.dtype)

        # BLOCK_N must be >= N and power of 2
        BLOCK_N = triton.next_power_of_2(N)

        gn_min_kernel[(M,)](
            gemm_out, self.group_norm.weight, self.group_norm.bias, row_min,
            M, N,
            NUM_GROUPS=self.num_groups,
            GROUP_SIZE=self.group_size,
            eps=self.eps,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )

        # row_min: [M], reshape to [1, 1, M, 1] then add bias [1, N, 1, 1]
        # min_x shape after torch.min keepdim: [M, 1]
        # then + bias [1, N, 1, 1] => [1, N, M, 1]
        min_x = row_min.view(M, 1)
        out = min_x + self.bias  # broadcast: [M,1] + [1,N,1,1] -> [1,N,M,1]
        return out