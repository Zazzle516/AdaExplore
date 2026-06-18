import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    SCALE_HALF: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm_mask = rm < M
    rn_mask = rn < N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)
        k_mask = rk < K

        x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
        x = tl.load(x_ptrs, mask=rm_mask[:, None] & k_mask[None, :], other=0.0)

        w_ptrs = w_ptr + rn[:, None] * stride_wn + rk[None, :] * stride_wk
        w = tl.load(w_ptrs, mask=rn_mask[:, None] & k_mask[None, :], other=0.0)

        acc += tl.dot(x, tl.trans(w))

    # apply scale * 0.5 and mask out invalid n columns before reducing
    acc = acc * SCALE_HALF
    acc = tl.where(rn_mask[None, :], acc, 0.0)
    row_sum = tl.sum(acc, axis=1)

    tl.atomic_add(out_ptr + rm, row_sum, mask=rm_mask)


def fused_matmul_div_sum_scale(x: torch.Tensor, w: torch.Tensor, scaling_factor: float):
    M, K = x.shape
    N, K2 = w.shape
    assert K == K2
    x = x.contiguous()
    w = w.contiguous()
    out = torch.zeros((M, 1), device=x.device, dtype=torch.float32)

    scale_half = float(scaling_factor) * 0.5

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    fused_kernel[grid](
        x, w, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        SCALE_HALF=scale_half,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        return fused_matmul_div_sum_scale(x, self.weight, self.scaling_factor)