import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 8}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def split_k_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_sk = tl.program_id(1)

    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    GROUP_M = 8
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K_per_split = tl.cdiv(K, SPLIT_K)
    num_iters = tl.cdiv(K_per_split, BLOCK_K)

    a_stride_iter = BLOCK_K * stride_ak
    b_stride_iter = BLOCK_K * stride_bk

    for i in range(0, num_iters):
        k_curr = pid_sk * BLOCK_K + i * BLOCK_K * 1
        # Compute actual k offset for this iter (we step by SPLIT_K*BLOCK_K within K dimension)
        # Actually we want each split-k program to read a contiguous chunk of K.
        # Let chunk_size = K_per_split, start = pid_sk * K_per_split.
        # We've initialized offs_k for the first block. Now we step by BLOCK_K.
        k_offset = pid_sk * K_per_split + i * BLOCK_K
        mask_k = (i * BLOCK_K + tl.arange(0, BLOCK_K)) < K_per_split
        mask_k_full = (k_offset + tl.arange(0, BLOCK_K)) < K
        mask = mask_k & mask_k_full
        a = tl.load(a_ptrs, mask=mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += a_stride_iter
        b_ptrs += b_stride_iter

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if SPLIT_K == 1:
        tl.store(c_ptrs, acc, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, acc, mask=c_mask)


@triton.jit
def epilogue_swish_kernel(
    c_ptr, bias_ptr, out_ptr,
    M, N,
    stride_cm, stride_cn,
    stride_om, stride_on,
    scaling_factor,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c = tl.load(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = c + bias[None, :]
    sig = tl.sigmoid(acc)
    out = acc * sig * scaling_factor
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             out.to(out_ptr.dtype.element_ty), mask=mask)


def linear_swish_scale(x_fp16, weight_t_fp16, bias, scaling_factor):
    M, K = x_fp16.shape
    K2, N = weight_t_fp16.shape
    assert K == K2

    # Accumulator buffer for split-k atomic adds
    acc_buf = torch.zeros((M, N), device=x_fp16.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        META['SPLIT_K'],
    )

    split_k_matmul_kernel[grid](
        x_fp16, weight_t_fp16, acc_buf,
        M, N, K,
        x_fp16.stride(0), x_fp16.stride(1),
        weight_t_fp16.stride(0), weight_t_fp16.stride(1),
        acc_buf.stride(0), acc_buf.stride(1),
    )

    out = torch.empty((M, N), device=x_fp16.device, dtype=torch.float16)
    BLOCK_M = 32
    BLOCK_N = 256
    grid2 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    epilogue_swish_kernel[grid2](
        acc_buf, bias, out,
        M, N,
        acc_buf.stride(0), acc_buf.stride(1),
        out.stride(0), out.stride(1),
        scaling_factor,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)
        self.in_features = in_features
        self.out_features = out_features
        with torch.no_grad():
            wt = self.matmul.weight.t().contiguous().to(torch.float16)
        self.register_buffer('weight_t_fp16', wt, persistent=False)

    def forward(self, x):
        x = x.contiguous().to(torch.float16)
        b = self.matmul.bias.contiguous().to(torch.float32)
        return linear_swish_scale(x, self.weight_t_fp16, b, self.scaling_factor)