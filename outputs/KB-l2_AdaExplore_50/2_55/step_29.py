import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 1024, 'GROUPS_PER_PROG': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 2048, 'GROUPS_PER_PROG': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 1024, 'GROUPS_PER_PROG': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 2048, 'GROUPS_PER_PROG': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 512, 'GROUPS_PER_PROG': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 4096, 'GROUPS_PER_PROG': 4}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_K': 512, 'GROUPS_PER_PROG': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 1024, 'GROUPS_PER_PROG': 32}, num_warps=8, num_stages=2),
    ],
    key=['IN_FEATURES', 'OUT_FEATURES'],
)
@triton.jit
def fused_kernel(
    x_ptr, w_ptr, b_ptr, partial_ptr,
    BATCH, IN_FEATURES, OUT_FEATURES, N_GROUPS,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUPS_PER_PROG: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    OUT_PER_PROG: tl.constexpr = GROUPS_PER_PROG * KERNEL_SIZE

    x_row_ptr = x_ptr + pid_b * IN_FEATURES
    base_out = pid_g * OUT_PER_PROG

    acc = tl.zeros((OUT_PER_PROG,), dtype=tl.float32)

    offs_out_local = tl.arange(0, OUT_PER_PROG)
    offs_out = base_out + offs_out_local
    out_mask = offs_out < OUT_FEATURES

    for k_start in range(0, IN_FEATURES, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < IN_FEATURES
        x_vals = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0)

        w_ptrs = w_ptr + offs_out[:, None] * IN_FEATURES + offs_k[None, :]
        w_mask = out_mask[:, None] & mask_k[None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

    b_vals = tl.load(b_ptr + offs_out, mask=out_mask, other=0.0)
    acc = acc + b_vals

    acc2d = tl.reshape(acc, (GROUPS_PER_PROG, KERNEL_SIZE))
    pooled = tl.max(acc2d, axis=1)

    partial_sum = tl.sum(pooled, axis=0)

    n_chunks = tl.num_programs(1)
    tl.store(partial_ptr + pid_b * n_chunks + pid_g, partial_sum)


@triton.jit
def finalize_kernel(
    partial_ptr, out_ptr,
    BATCH, N_CHUNKS,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    row_ptr = partial_ptr + pid_b * N_CHUNKS
    offs = tl.arange(0, BLOCK)
    mask = offs < N_CHUNKS
    v = tl.load(row_ptr + offs, mask=mask, other=0.0)
    s = tl.sum(v, axis=0) * SCALE
    tl.store(out_ptr + pid_b, s)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = float(scale_factor)
        self.matmul = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()
        device = x.device
        W = self.matmul.weight.to(device).contiguous()
        b = self.matmul.bias.to(device).contiguous()
        B = x.shape[0]

        assert self.out_features % self.kernel_size == 0
        n_groups = self.out_features // self.kernel_size

        # We need to know n_chunks based on chosen GROUPS_PER_PROG.
        # Allocate partials with max possible size; we'll size per autotune.
        # Use a single allocation big enough; rely on grid lambda to set n_chunks.
        # Allocate inside grid lambda is not possible, so allocate worst case (smallest GPP).
        max_chunks = triton.cdiv(n_groups, 4)  # smallest GPP in configs is 4
        partial = torch.empty((B, max_chunks), device=device, dtype=torch.float32)
        out = torch.empty(B, device=device, dtype=torch.float32)

        # Hold actual n_chunks via a mutable container.
        actual = {}

        def grid1(meta):
            gpp = meta['GROUPS_PER_PROG']
            nc = triton.cdiv(n_groups, gpp)
            actual['nc'] = nc
            return (B, nc)

        fused_kernel[grid1](
            x, W, b, partial,
            B, self.in_features, self.out_features, n_groups,
            KERNEL_SIZE=self.kernel_size,
        )

        nc = actual['nc']
        # pick BLOCK as next power of 2 >= nc
        block = 1
        while block < nc:
            block *= 2
        block = max(block, 16)

        finalize_kernel[(B,)](
            partial, out,
            B, nc,
            SCALE=self.scale_factor,
            BLOCK=block,
        )
        return out