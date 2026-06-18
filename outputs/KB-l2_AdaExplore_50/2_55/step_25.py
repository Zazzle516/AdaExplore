import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 1024, 'GROUPS_PER_PROG': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 2048, 'GROUPS_PER_PROG': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 2048, 'GROUPS_PER_PROG': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 4096, 'GROUPS_PER_PROG': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 4096, 'GROUPS_PER_PROG': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 1024, 'GROUPS_PER_PROG': 16}, num_warps=4, num_stages=3),
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
    # Each program handles one batch row and a tile of GROUPS_PER_PROG groups.
    # Computes the sum of pooled values over its group tile, writes a single
    # value into partial[batch, prog_g] for deterministic stage-2 reduction.
    pid_b = tl.program_id(0)
    pid_gtile = tl.program_id(1)

    OUT_PER_PROG: tl.constexpr = GROUPS_PER_PROG * KERNEL_SIZE

    x_row_ptr = x_ptr + pid_b * IN_FEATURES
    base_out = pid_gtile * OUT_PER_PROG

    offs_out = base_out + tl.arange(0, OUT_PER_PROG)  # [OUT_PER_PROG]
    out_mask = offs_out < OUT_FEATURES

    # accumulator per output row in tile
    acc = tl.zeros((OUT_PER_PROG,), dtype=tl.float32)

    for k_start in range(0, IN_FEATURES, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < IN_FEATURES
        x_vals = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0)  # [BLOCK_K]

        w_ptrs = w_ptr + offs_out[:, None] * IN_FEATURES + offs_k[None, :]
        w_mask = out_mask[:, None] & mask_k[None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [OUT_PER_PROG, BLOCK_K]

        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

    # bias
    b_vals = tl.load(b_ptr + offs_out, mask=out_mask, other=0.0)
    acc = acc + b_vals

    # reshape conceptually: [GROUPS_PER_PROG, KERNEL_SIZE] then max over kernel
    acc2 = tl.reshape(acc, (GROUPS_PER_PROG, KERNEL_SIZE))
    pooled = tl.max(acc2, axis=1)  # [GROUPS_PER_PROG]

    # mask groups outside valid range
    g_offs = pid_gtile * GROUPS_PER_PROG + tl.arange(0, GROUPS_PER_PROG)
    g_mask = g_offs < N_GROUPS
    pooled = tl.where(g_mask, pooled, 0.0)

    s = tl.sum(pooled, axis=0)
    tl.store(partial_ptr + pid_b * tl.num_programs(1) + pid_gtile, s)


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
        n_groups = self.out_features // self.kernel_size

        # We'll launch with a config-determined GROUPS_PER_PROG; allocate
        # partial buffer with worst-case size. Use a fixed launch-time block size.
        # Determine via autotune by making partial size = ceil(n_groups / gpp).
        # Since gpp varies, we allocate enough for the smallest gpp we use (8).
        MIN_GPP = 8
        max_gtiles = (n_groups + MIN_GPP - 1) // MIN_GPP
        partial = torch.zeros((B, max_gtiles), device=x.device, dtype=torch.float32)

        def grid(meta):
            gpp = meta['GROUPS_PER_PROG']
            return (B, (n_groups + gpp - 1) // gpp)

        fused_kernel[grid](
            x, W, b, partial,
            B, self.in_features, self.out_features, n_groups,
            KERNEL_SIZE=self.kernel_size,
        )

        out = partial.sum(dim=1) * self.scale_factor
        return out