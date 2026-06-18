import torch
import torch.nn as nn
import triton
import triton.language as tl


AUTOTUNE_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_awt_mul_lrelu_kernel(
    A_ptr, W_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    MULTIPLIER: tl.constexpr, NEG_SLOPE: tl.constexpr,
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

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # A: (M, K) row-major -> A[m, k]
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # W: (N, K) row-major -> W[n, k]  (we will use tl.dot(a, w_tile.T))
    w_ptrs = W_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, tl.trans(w), input_precision='tf32')
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    offs_m_out = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_out = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m_out < M
    mask_n = offs_n_out < N

    bias = tl.load(bias_ptr + offs_n_out, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]
    acc = acc * MULTIPLIER
    acc = tl.where(acc >= 0, acc, acc * NEG_SLOPE)

    c_ptrs = C_ptr + (offs_m_out[:, None] * stride_cm + offs_n_out[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(mask_m[:, None]) & (mask_n[None, :]))


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = float(multiplier)
        self.negative_slope = float(negative_slope)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.gemm.weight  # (out, in), row-major
        b = self.gemm.bias    # (out,)

        M, K = x.shape
        N = W.shape[0]

        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        )

        gemm_awt_mul_lrelu_kernel[grid](
            x, W, b, out,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            out.stride(0), out.stride(1),
            MULTIPLIER=self.multiplier, NEG_SLOPE=self.negative_slope,
        )
        return out