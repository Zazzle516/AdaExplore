import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
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

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
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
    M, N: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    eps: tl.constexpr,
):
    pid = tl.program_id(0)

    # Load full row [N] -> view as [NUM_GROUPS, GROUP_SIZE]
    row_offs = tl.arange(0, N)
    x = tl.load(X_ptr + pid * N + row_offs)
    w = tl.load(W_ptr + row_offs)
    b = tl.load(B_ptr + row_offs)

    x2d = tl.reshape(x, (NUM_GROUPS, GROUP_SIZE))
    w2d = tl.reshape(w, (NUM_GROUPS, GROUP_SIZE))
    b2d = tl.reshape(b, (NUM_GROUPS, GROUP_SIZE))

    inv_gs = 1.0 / GROUP_SIZE
    sum_x = tl.sum(x2d, axis=1)        # [NUM_GROUPS]
    sum_x2 = tl.sum(x2d * x2d, axis=1) # [NUM_GROUPS]
    mean = sum_x * inv_gs
    var = sum_x2 * inv_gs - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    norm = (x2d - mean[:, None]) * rstd[:, None]
    y = norm * w2d + b2d  # [NUM_GROUPS, GROUP_SIZE]

    # Min over entire row
    y_flat = tl.reshape(y, (N,))
    row_min = tl.min(y_flat, axis=0)

    tl.store(OUT_ptr + pid, row_min)


def triton_gemm(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_kernel[grid](
        x, weight, out, bias,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
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

        gemm_out = triton_gemm(x, self.gemm.weight, self.gemm.bias)

        row_min = torch.empty((M,), device=x.device, dtype=x.dtype)

        gn_min_kernel[(M,)](
            gemm_out, self.group_norm.weight, self.group_norm.bias, row_min,
            M, N,
            NUM_GROUPS=self.num_groups,
            GROUP_SIZE=self.group_size,
            eps=self.eps,
            num_warps=16,
            num_stages=3,
        )

        min_x = row_min.view(M, 1)
        out = min_x + self.bias
        return out