import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 512, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_K': 512, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_K': 1024, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_K': 1024, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 256, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_K': 256, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_K': 2048, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
    ],
    key=['IN_FEATURES', 'OUT_FEATURES', 'K_SPLITS'],
)
@triton.jit
def fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    BATCH, IN_FEATURES, OUT_FEATURES,
    K_SPLITS: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)

    # Per-program K range
    k_per_split = IN_FEATURES // K_SPLITS
    k_lo = pid_k * k_per_split
    k_hi = k_lo + k_per_split

    x_row_ptr = x_ptr + pid_b * IN_FEATURES

    # scalar accumulator for the final reduced value
    scalar_acc = tl.zeros((1,), dtype=tl.float32)

    offs_k_base = tl.arange(0, BLOCK_K)
    offs_n_base = tl.arange(0, BLOCK_N)

    # Loop over N tiles
    for n_start in range(0, OUT_FEATURES, BLOCK_N):
        offs_n = n_start + offs_n_base
        n_mask = offs_n < OUT_FEATURES

        # Partial GEMM accumulator for this N tile
        n_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k_start in range(k_lo, k_hi, BLOCK_K):
            offs_k = k_start + offs_k_base
            k_mask = offs_k < IN_FEATURES

            x_vals = tl.load(x_row_ptr + offs_k, mask=k_mask, other=0.0)  # [BLOCK_K]

            w_ptrs = w_ptr + offs_n[:, None] * IN_FEATURES + offs_k[None, :]
            w_mask = n_mask[:, None] & k_mask[None, :]
            w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_N, BLOCK_K]

            n_acc += tl.sum(w_vals * x_vals[None, :], axis=1)

        # Only one program adds the bias (pid_k == 0)
        if pid_k == 0:
            b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
            n_acc = n_acc + b_vals

        # Reshape to [BLOCK_N/KERNEL_SIZE, KERNEL_SIZE], max over kernel, sum
        n_acc2d = tl.reshape(n_acc, (BLOCK_N // KERNEL_SIZE, KERNEL_SIZE))
        pooled = tl.max(n_acc2d, axis=1)
        scalar_acc += tl.sum(pooled, axis=0)

    val = tl.sum(scalar_acc, axis=0)
    if K_SPLITS == 1:
        val = val * SCALE
        tl.store(out_ptr + pid_b, val)
    else:
        tl.atomic_add(out_ptr + pid_b, val)


@triton.jit
def scale_kernel(out_ptr, BATCH, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < BATCH
    v = tl.load(out_ptr + offs, mask=mask, other=0.0)
    v = v * SCALE
    tl.store(out_ptr + offs, v, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = float(scale_factor)
        self.matmul = nn.Linear(in_features, out_features)

        # Choose K_SPLITS to balance occupancy. Batch=128, SMs=128 on 4090.
        # With B=128, splits=1 already fills 128 programs. Use splits=1.
        # But we can try splits=2 for more parallelism on K.
        self.K_SPLITS = 1

    def forward(self, x):
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()
        device = x.device
        W = self.matmul.weight.to(device).contiguous()
        b = self.matmul.bias.to(device).contiguous()
        B = x.shape[0]

        K_SPLITS = self.K_SPLITS
        # ensure divisibility
        if self.in_features % K_SPLITS != 0:
            K_SPLITS = 1

        if K_SPLITS == 1:
            out = torch.empty(B, device=device, dtype=torch.float32)
        else:
            out = torch.zeros(B, device=device, dtype=torch.float32)

        grid = (B, K_SPLITS)
        fused_kernel[grid](
            x, W, b, out,
            B, self.in_features, self.out_features,
            K_SPLITS=K_SPLITS,
            KERNEL_SIZE=self.kernel_size,
            SCALE=self.scale_factor,
        )

        if K_SPLITS > 1:
            BLOCK = 128
            grid2 = ((B + BLOCK - 1) // BLOCK,)
            scale_kernel[grid2](out, B, SCALE=self.scale_factor, BLOCK=BLOCK)

        return out