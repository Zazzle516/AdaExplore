import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _reduce_w_kernel(
    w_ptr, wsum_ptr,
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
        # W shape (H, K): row h, col k
        ptrs = w_ptr + offs_h[:, None] * stride_wh + offs_k[None, :] * stride_wk
        mask = mask_h[:, None] & mask_k[None, :]
        vals = tl.load(ptrs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)

    tl.store(wsum_ptr + offs_k, acc, mask=mask_k)


@triton.jit
def _dot_kernel(
    x_ptr, wsum_ptr, out_ptr,
    M, K,
    stride_xm, stride_xk,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        mask = mask_m[:, None] & mask_k[None, :]
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
        w_vals = tl.load(wsum_ptr + offs_k, mask=mask_k, other=0.0)
        prod = x_vals * w_vals[None, :]
        acc += tl.sum(prod, axis=1)

    acc = acc * SCALE
    tl.store(out_ptr + offs_m, acc, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = float(scaling_factor)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()
        W = self.weight
        if not W.is_cuda:
            W = W.cuda()

        M, K = x.shape
        H = W.shape[0]
        assert K == self.input_size and H == self.hidden_size

        wsum = torch.empty(K, device=x.device, dtype=torch.float32)

        BLOCK_K_R = 128
        BLOCK_H_R = 64
        grid_r = ((K + BLOCK_K_R - 1) // BLOCK_K_R,)
        _reduce_w_kernel[grid_r](
            W, wsum,
            H, K,
            W.stride(0), W.stride(1),
            BLOCK_K=BLOCK_K_R,
            BLOCK_H=BLOCK_H_R,
            num_warps=4, num_stages=2,
        )

        out = torch.empty((M, 1), device=x.device, dtype=x.dtype)

        BLOCK_M = 64
        BLOCK_K_D = 256
        grid_d = ((M + BLOCK_M - 1) // BLOCK_M,)
        # scale = 1/2 * scaling_factor
        SCALE = 0.5 * self.scaling_factor
        _dot_kernel[grid_d](
            x, wsum, out,
            M, K,
            x.stride(0), x.stride(1),
            SCALE=SCALE,
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K_D,
            num_warps=4, num_stages=2,
        )

        return out