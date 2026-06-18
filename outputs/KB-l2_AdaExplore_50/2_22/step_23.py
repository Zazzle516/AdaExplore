import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


# GEMM kernel with fused bias + scale*2 + clamp
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_fused_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    SCALE2: tl.constexpr,
    CMIN: tl.constexpr,
    CMAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        mask_k = k_offs < K
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    b = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]
    acc = acc * SCALE2
    acc = tl.minimum(tl.maximum(acc, CMIN), CMAX)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


# Online single-pass LSE + mish kernel.
# Reads each row of the [M, N] tensor once.
@triton.jit
def online_lse_mish_kernel(
    in_ptr,        # [M, N]
    out_ptr,       # [M, 1]
    M, N,
    stride_im, stride_in,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)

    m_val = -float('inf')
    s_val = 0.0

    for n_start in range(0, N, BLOCK_N):
        idx = n_start + offs_n
        mask = idx < N
        v = tl.load(in_ptr + pid_m * stride_im + idx * stride_in,
                    mask=mask, other=-float('inf'))

        block_max = tl.max(v, axis=0)
        new_m = tl.maximum(m_val, block_max)
        # rescale running sum
        s_val = s_val * tl.exp(m_val - new_m)
        e = tl.exp(v - new_m)
        e = tl.where(mask, e, 0.0)
        s_val = s_val + tl.sum(e, axis=0)
        m_val = new_m

    lse = m_val + tl.log(s_val)

    # mish: x * tanh(softplus(x))
    sp = tl.where(lse > 0, lse, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(lse)))
    e2 = tl.exp(2.0 * sp)
    tanh_sp = (e2 - 1.0) / (e2 + 1.0)
    result = lse * lse * tanh_sp

    tl.store(out_ptr + pid_m, result)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super().__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = float(scale_factor)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.matmul.weight.contiguous().cuda()
        b = self.matmul.bias.contiguous().cuda()

        M, K = x.shape
        N = W.shape[0]

        gemm_out = torch.empty((M, N), device=x.device, dtype=torch.float32)

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
        )
        gemm_fused_kernel[grid](
            x, W, b, gemm_out,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            gemm_out.stride(0), gemm_out.stride(1),
            SCALE2=self.scale_factor * 2.0,
            CMIN=self.clamp_min,
            CMAX=self.clamp_max,
        )

        out = torch.empty((M, 1), device=x.device, dtype=torch.float32)
        BLOCK_N = 1024
        online_lse_mish_kernel[(M,)](
            gemm_out, out,
            M, N,
            gemm_out.stride(0), gemm_out.stride(1),
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )

        return out