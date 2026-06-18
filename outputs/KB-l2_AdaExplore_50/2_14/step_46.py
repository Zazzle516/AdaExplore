import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gemm_rowsum_kernel(
    x_ptr, w_ptr, out_ptr,
    M, H, K,
    stride_xm, stride_xk,
    stride_wh, stride_wk,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k_base = tl.arange(0, BLOCK_K)

    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for n_start in range(0, H, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k_base[None, :] * stride_xk
        # W is (H, K); we want (K, N) tile for tl.dot, so index W[n, k] -> w_ptr + n*stride_wh + k*stride_wk
        w_ptrs = w_ptr + offs_k_base[:, None] * stride_wk + offs_n[None, :] * stride_wh
        for k_start in range(0, K, BLOCK_K):
            x_vals = tl.load(x_ptrs)
            w_vals = tl.load(w_ptrs)
            acc += tl.dot(x_vals, w_vals)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk
        row_sum += tl.sum(acc, axis=1)

    row_sum = row_sum * SCALE
    tl.store(out_ptr + offs_m, row_sum)


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

        out = torch.empty((M, 1), device=x.device, dtype=x.dtype)

        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32
        assert M % BLOCK_M == 0 and H % BLOCK_N == 0 and K % BLOCK_K == 0
        grid = (M // BLOCK_M,)
        SCALE = 0.5 * self.scaling_factor
        _gemm_rowsum_kernel[grid](
            x, W, out,
            M, H, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            SCALE=SCALE,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=3,
        )

        return out