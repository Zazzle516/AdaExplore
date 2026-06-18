import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 32, 'BLOCK_H': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_B': 64, 'BLOCK_H': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_B': 32, 'BLOCK_H': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_B': 64, 'BLOCK_H': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_B': 128, 'BLOCK_H': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_B': 32, 'BLOCK_H': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
    ],
    key=['B', 'H', 'K'],
)
@triton.jit
def fused_matmul_sum_kernel(
    x_ptr, w_ptr, out_ptr,
    B, H, K,
    stride_xb, stride_xk,
    stride_wh, stride_wk,
    stride_ob,
    scale,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_k = tl.arange(0, BLOCK_K)

    mask_b = offs_b < B
    mask_h = offs_h < H

    acc = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)

    x_ptrs = x_ptr + offs_b[:, None] * stride_xb + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_h[:, None] * stride_wh + offs_k[None, :] * stride_wk

    for k_start in range(0, K, BLOCK_K):
        k_mask = (k_start + offs_k) < K
        x = tl.load(x_ptrs + k_start * stride_xk,
                    mask=mask_b[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs + k_start * stride_wk,
                    mask=mask_h[:, None] & k_mask[None, :], other=0.0)
        # w is (BLOCK_H, BLOCK_K); we want x @ w.T -> (BLOCK_B, BLOCK_H)
        acc += tl.dot(x, tl.trans(w))

    # row-reduce along H slice, fold (/2) * scale
    row_partial = tl.sum(acc * mask_h[None, :].to(tl.float32), axis=1) * (scale * 0.5)

    # atomic add to out[offs_b]
    tl.atomic_add(out_ptr + offs_b, row_partial, mask=mask_b)


def fused_matmul_div_sum_scale(x: torch.Tensor, w: torch.Tensor, scale: float) -> torch.Tensor:
    B, K = x.shape
    H, K2 = w.shape
    assert K == K2
    x = x.contiguous()
    w = w.contiguous()
    out = torch.zeros((B,), device=x.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_B']), triton.cdiv(H, meta['BLOCK_H']))
    fused_matmul_sum_kernel[grid](
        x, w, out,
        B, H, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0),
        scale,
    )
    return out.view(B, 1)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        x = x.cuda() if not x.is_cuda else x
        return fused_matmul_div_sum_scale(x, self.weight, self.scaling_factor)