import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['IN_FEATURES', 'OUT_FEATURES'],
)
@triton.jit
def fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    BATCH, IN_FEATURES, OUT_FEATURES,
    KERNEL_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < BATCH
    mask_n = offs_n < OUT_FEATURES

    x_ptrs = x_ptr + offs_m[:, None] * IN_FEATURES + offs_k[None, :]
    # W is stored as [OUT, IN]; load tile as [BLOCK_N, BLOCK_K] then trans.
    w_ptrs = w_ptr + offs_n[:, None] * IN_FEATURES + offs_k[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, IN_FEATURES, BLOCK_K):
        k_remaining = IN_FEATURES - k_start
        mask_k = offs_k < k_remaining
        x_vals = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w_vals = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc += tl.dot(x_vals, tl.trans(w_vals), allow_tf32=True)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    # bias
    b_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b_vals[None, :]

    # zero out invalid output positions so they don't pollute max
    very_neg = tl.full((BLOCK_M, BLOCK_N), -1e30, dtype=tl.float32)
    acc = tl.where(mask_n[None, :], acc, very_neg)

    # max pool along N with KERNEL_SIZE
    BN_POOLED: tl.constexpr = BLOCK_N // KERNEL_SIZE
    acc2 = tl.reshape(acc, (BLOCK_M, BN_POOLED, KERNEL_SIZE))
    pooled = tl.max(acc2, axis=2)  # [BLOCK_M, BN_POOLED]

    # mask pooled positions outside valid range
    pooled_n = pid_n * BN_POOLED + tl.arange(0, BN_POOLED)
    pooled_mask = pooled_n < (OUT_FEATURES // KERNEL_SIZE)
    pooled = tl.where(pooled_mask[None, :], pooled, 0.0)

    row_sum = tl.sum(pooled, axis=1)  # [BLOCK_M]
    row_sum = row_sum * SCALE

    tl.atomic_add(out_ptr + offs_m, row_sum, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = float(scale_factor)
        self.matmul = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.matmul.weight.to(x.device).contiguous()
        b = self.matmul.bias.to(x.device).contiguous()
        B = x.shape[0]

        assert self.out_features % self.kernel_size == 0

        out = torch.zeros(B, device=x.device, dtype=torch.float32)

        def grid(meta):
            return (
                triton.cdiv(B, meta['BLOCK_M']),
                triton.cdiv(self.out_features, meta['BLOCK_N']),
            )

        fused_kernel[grid](
            x, W, b, out,
            B, self.in_features, self.out_features,
            KERNEL_SIZE=self.kernel_size,
            SCALE=self.scale_factor,
        )

        return out