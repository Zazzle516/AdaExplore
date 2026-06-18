import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 2048}, num_warps=8, num_stages=3),
    ],
    key=['IN_FEATURES', 'OUT_FEATURES'],
)
@triton.jit
def fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    BATCH, IN_FEATURES, OUT_FEATURES,
    KERNEL_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # one program per (batch row, group of KERNEL_SIZE output features that produce one pooled value)
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)  # group index, total groups = OUT_FEATURES // KERNEL_SIZE

    # We compute, for each of KERNEL_SIZE output features in this group:
    #   acc[k] = sum_i x[pid_b, i] * w[pid_g*KS + k, i] + b[pid_g*KS + k]
    # Then pooled = max over k, and the program contributes pooled to the row sum.

    x_row_ptr = x_ptr + pid_b * IN_FEATURES
    base_out = pid_g * KERNEL_SIZE

    # accumulator per kernel position
    acc = tl.zeros((KERNEL_SIZE,), dtype=tl.float32)

    for k_start in range(0, IN_FEATURES, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < IN_FEATURES
        x_vals = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Load weight rows for each of KERNEL_SIZE outputs in group
        # w shape: [OUT_FEATURES, IN_FEATURES]
        offs_out = base_out + tl.arange(0, KERNEL_SIZE)  # [KERNEL_SIZE]
        w_ptrs = w_ptr + offs_out[:, None] * IN_FEATURES + offs_k[None, :]
        w_vals = tl.load(w_ptrs, mask=mask_k[None, :], other=0.0)  # [KS, BLOCK_K]

        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

    # add bias
    offs_out = base_out + tl.arange(0, KERNEL_SIZE)
    b_vals = tl.load(b_ptr + offs_out)
    acc = acc + b_vals

    # max pool over kernel_size
    pooled = tl.max(acc, axis=0)
    pooled = pooled * SCALE

    # atomic add to out[pid_b]
    tl.atomic_add(out_ptr + pid_b, pooled)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = float(scale_factor)
        # Use nn.Linear for parameter init compatibility
        self.matmul = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.matmul.weight.contiguous()
        b = self.matmul.bias.contiguous()
        B = x.shape[0]
        out = torch.zeros(B, device=x.device, dtype=torch.float32)

        assert self.out_features % self.kernel_size == 0
        n_groups = self.out_features // self.kernel_size

        grid = (B, n_groups)
        fused_kernel[grid](
            x, W, b, out,
            B, self.in_features, self.out_features,
            KERNEL_SIZE=self.kernel_size,
            SCALE=self.scale_factor,
        )
        return out