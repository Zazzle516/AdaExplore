import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=4, num_stages=2),
    ],
    key=['K', 'N'],
)
@triton.jit
def fused_linear_sigmoid_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program per row of M
    pid_m = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    row_sum = tl.zeros([], dtype=tl.float32)

    # Loop over N tiles
    for n_start in range(0, N, BLOCK_N):
        cur_n = n_start + offs_n  # [BLOCK_N]
        n_mask = cur_n < N

        # Accumulator for this N tile: [BLOCK_N]
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)

        # Loop over K dimension
        for k_start in range(0, K, BLOCK_K):
            cur_k = k_start + offs_k  # [BLOCK_K]
            k_mask = cur_k < K

            # Load x[pid_m, k_start:k_start+BLOCK_K] -> [BLOCK_K]
            x_vals = tl.load(
                x_ptr + pid_m * stride_xm + cur_k * stride_xk,
                mask=k_mask, other=0.0,
            )  # [BLOCK_K]

            # Load w[cur_n, cur_k] -> [BLOCK_N, BLOCK_K]
            w_ptrs = w_ptr + cur_n[:, None] * stride_wn + cur_k[None, :] * stride_wk
            w_vals = tl.load(
                w_ptrs,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            )  # [BLOCK_N, BLOCK_K]

            # acc += w_vals @ x_vals  -> [BLOCK_N]
            acc += tl.sum(w_vals * x_vals[None, :], axis=1)

        # Add bias
        bias = tl.load(b_ptr + cur_n, mask=n_mask, other=0.0)
        acc += bias

        # Sigmoid
        sig = tl.sigmoid(acc)
        sig = tl.where(n_mask, sig, 0.0)

        # Sum across this tile and accumulate into scalar
        row_sum += tl.sum(sig, axis=0)

    # Store
    tl.store(out_ptr + pid_m, row_sum)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        x = x.contiguous().cuda()
        M, K = x.shape
        N = self.hidden_size

        weight = self.linear.weight.contiguous()  # [N, K]
        bias = self.linear.bias.contiguous()      # [N]

        out = torch.empty((M, 1), device=x.device, dtype=torch.float32)

        grid = (M,)
        fused_linear_sigmoid_sum_kernel[grid](
            x, weight, bias, out,
            M, N, K,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
        )
        return out