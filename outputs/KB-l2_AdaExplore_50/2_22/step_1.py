import torch
import torch.nn as nn
import triton
import triton.language as tl


# Fused GEMM + bias + scale*2 + clamp + LSE (per row) + mish epilogue
# Output per row: y = lse * mish(lse) where lse = logsumexp(clamp(2*scale*(x@W^T + b)))

GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=2, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_partial_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    SCALE2: tl.constexpr,
    CLAMP_MIN: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remain = K - k
        mask_k = offs_k < k_remain
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]
    # scale * 2 (because x = x*scale then x+x => 2*scale)
    acc = acc * SCALE2
    acc = tl.minimum(tl.maximum(acc, CLAMP_MIN), CLAMP_MAX)

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def lse_mish_kernel(
    in_ptr, out_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    row_ptr = in_ptr + pid * N

    # Two-pass: max then sum exp
    max_val = -float('inf')
    for n_start in range(0, N, BLOCK_N):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        v = tl.load(row_ptr + offs, mask=mask, other=-float('inf'))
        m_blk = tl.max(v, axis=0)
        max_val = tl.maximum(max_val, m_blk)

    sum_exp = 0.0
    for n_start in range(0, N, BLOCK_N):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        v = tl.load(row_ptr + offs, mask=mask, other=-float('inf'))
        e = tl.exp(v - max_val)
        e = tl.where(mask, e, 0.0)
        sum_exp += tl.sum(e, axis=0)

    lse = max_val + tl.log(sum_exp)
    # mish(lse) = lse * tanh(softplus(lse))
    sp = tl.log(1.0 + tl.exp(lse))
    # tanh via exp
    e2 = tl.exp(2.0 * sp)
    tanh_sp = (e2 - 1.0) / (e2 + 1.0)
    mish_lse = lse * tanh_sp
    out_val = lse * mish_lse
    tl.store(out_ptr + pid, out_val)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.scale_factor = float(scale_factor)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.matmul = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.matmul.weight.contiguous()
        b = self.matmul.bias.contiguous()
        M, K = x.shape
        N = W.shape[0]

        partial = torch.empty((M, N), device=x.device, dtype=torch.float32)

        scale2 = self.scale_factor * 2.0

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
        gemm_partial_kernel[grid](
            x, W, b, partial,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            SCALE2=scale2,
            CLAMP_MIN=self.clamp_min,
            CLAMP_MAX=self.clamp_max,
        )

        out = torch.empty((M, 1), device=x.device, dtype=torch.float32)
        BLOCK_N = 1024
        lse_mish_kernel[(M,)](partial, out, M, N, BLOCK_N=BLOCK_N, num_warps=8)
        return out