import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['N', 'K'],
)
@triton.jit
def row_gemm_max_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # one program per row m
    pid_m = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    # base pointers for this row
    x_row_ptr = x_ptr + pid_m * stride_xm

    row_max = -float('inf')

    for n_start in range(0, N, BLOCK_N):
        cur_n = n_start + offs_n
        n_mask = cur_n < N

        # accumulator for this n-tile: BLOCK_N
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            cur_k = k_start + offs_k
            k_mask = cur_k < K

            # load x[m, k_start:k_start+BLOCK_K] -> [BLOCK_K]
            x_vals = tl.load(x_row_ptr + cur_k * stride_xk, mask=k_mask, other=0.0)

            # load w[n_start:n_start+BLOCK_N, k_start:k_start+BLOCK_K] -> [BLOCK_N, BLOCK_K]
            w_ptrs = w_ptr + cur_n[:, None] * stride_wn + cur_k[None, :] * stride_wk
            w_mask = n_mask[:, None] & k_mask[None, :]
            w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

            # acc += w_vals @ x_vals
            acc += tl.sum(w_vals * x_vals[None, :], axis=1)

        # add bias
        b_vals = tl.load(b_ptr + cur_n, mask=n_mask, other=-float('inf'))
        acc = acc + b_vals
        acc = tl.where(n_mask, acc, -float('inf'))

        tile_max = tl.max(acc, axis=0)
        row_max = tl.maximum(row_max, tile_max)

    # subtract mean over dim=1 of (B,1) -> 0; gelu(0) = 0
    diff = row_max - row_max  # = 0
    inv_sqrt2 = 0.70710678118654752440
    y = 0.5 * diff * (1.0 + tl.erf(diff * inv_sqrt2))
    tl.store(out_ptr + pid_m, y)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.gemm.weight.contiguous().cuda()
        bias = self.gemm.bias.contiguous().cuda()

        if self.max_dim == 1:
            M, K = x.shape
            N = weight.shape[0]
            out = torch.empty((M, 1), device=x.device, dtype=torch.float32)
            grid = (M,)
            row_gemm_max_kernel[grid](
                x, weight, bias, out,
                N, K,
                x.stride(0), x.stride(1),
                weight.stride(0), weight.stride(1),
            )
            return out
        else:
            y = torch.addmm(bias, x, weight.t())
            y = torch.max(y, dim=0, keepdim=True).values
            y = y - y.mean(dim=1, keepdim=True)
            inv_sqrt2 = 0.70710678118654752440
            out = 0.5 * y * (1.0 + torch.erf(y * inv_sqrt2))
            return out