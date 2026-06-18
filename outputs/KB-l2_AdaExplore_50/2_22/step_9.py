import torch
import torch.nn as nn
import triton
import triton.language as tl


# Fused GEMM + bias + scale*2 + clamp + online LSE reduction across N.
# Each program owns BLOCK_M rows and streams across the full N dimension,
# accumulating GEMM tile-by-tile while maintaining running (max, sum_exp)
# in registers. At the end, compute lse, apply mish, and store the (M,1) output.

GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def fused_gemm_lse_mish_kernel(
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M

    # Running max and sum_exp per row (initialized for online LSE)
    row_max = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Loop over N in BLOCK_N chunks
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

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
        acc = acc * SCALE2
        acc = tl.minimum(tl.maximum(acc, CLAMP_MIN), CLAMP_MAX)

        # Mask invalid N positions with -inf so they don't contribute
        acc = tl.where(mask_n[None, :], acc, -float('inf'))

        # Online LSE update
        tile_max = tl.max(acc, axis=1)  # [BLOCK_M]
        new_max = tl.maximum(row_max, tile_max)
        # Rescale previous sum
        alpha = tl.exp(row_max - new_max)
        # Compute current tile contribution
        e = tl.exp(acc - new_max[:, None])
        tile_sum = tl.sum(e, axis=1)
        row_sum = row_sum * alpha + tile_sum
        row_max = new_max

    lse = row_max + tl.log(row_sum)
    # mish(lse) = lse * tanh(softplus(lse))
    sp = tl.log(1.0 + tl.exp(lse))
    e2 = tl.exp(2.0 * sp)
    tanh_sp = (e2 - 1.0) / (e2 + 1.0)
    mish_lse = lse * tanh_sp
    out_val = lse * mish_lse

    tl.store(out_ptr + offs_m, out_val, mask=mask_m)


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

        scale2 = self.scale_factor * 2.0

        out = torch.empty((M, 1), device=x.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
        fused_gemm_lse_mish_kernel[grid](
            x, W, b, out,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            SCALE2=scale2,
            CLAMP_MIN=self.clamp_min,
            CLAMP_MAX=self.clamp_max,
        )
        return out