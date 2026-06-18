import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def col_sum_kernel(
    w_ptr, out_ptr,
    H, K,
    stride_wh, stride_wk,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    k_start = pid * BLOCK_K
    offs_k = k_start + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for h_start in range(0, H, BLOCK_H):
        offs_h = h_start + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        # W is (H, K), load tile (BLOCK_H, BLOCK_K)
        ptrs = w_ptr + offs_h[:, None] * stride_wh + offs_k[None, :] * stride_wk
        mask = mask_h[:, None] & mask_k[None, :]
        vals = tl.load(ptrs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)

    tl.store(out_ptr + offs_k, acc, mask=mask_k)


@triton.jit
def fused_dot_kernel(
    x_ptr, wcol_ptr, out_ptr,
    M, K,
    stride_xm, stride_xk,
    SCALE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    x_row_ptr = x_ptr + pid * stride_xm

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_vals = tl.load(x_row_ptr + offs_k * stride_xk, mask=mask_k, other=0.0)
        w_vals = tl.load(wcol_ptr + offs_k, mask=mask_k, other=0.0)
        acc += x_vals * w_vals

    total = tl.sum(acc, axis=0) * SCALE
    tl.store(out_ptr + pid, total)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = float(scaling_factor)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        x = x.contiguous()
        W = self.weight.contiguous()
        M, K = x.shape
        H = W.shape[0]
        assert K == W.shape[1]

        # Effective scale: x @ W.T -> /2 -> sum -> *scaling_factor
        # sum_n sum_k x[m,k]*W[n,k] / 2 * s = (s/2) * sum_k x[m,k] * sum_n W[n,k]
        eff_scale = self.scaling_factor * 0.5

        wcol = torch.empty(K, device=x.device, dtype=torch.float32)

        BLOCK_K1 = 128
        BLOCK_H = 128
        grid1 = ((K + BLOCK_K1 - 1) // BLOCK_K1,)
        col_sum_kernel[grid1](
            W, wcol,
            H, K,
            W.stride(0), W.stride(1),
            BLOCK_K=BLOCK_K1,
            BLOCK_H=BLOCK_H,
            num_warps=4,
        )

        out = torch.empty(M, device=x.device, dtype=torch.float32)
        BLOCK_K2 = 1024
        grid2 = (M,)
        fused_dot_kernel[grid2](
            x, wcol, out,
            M, K,
            x.stride(0), x.stride(1),
            SCALE=eff_scale,
            BLOCK_K=BLOCK_K2,
            num_warps=8,
        )
        return out.view(M, 1)