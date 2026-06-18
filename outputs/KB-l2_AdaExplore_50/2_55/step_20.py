import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 1024, 'GROUPS_PER_PROG': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 2048, 'GROUPS_PER_PROG': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 2048, 'GROUPS_PER_PROG': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 4096, 'GROUPS_PER_PROG': 4}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_K': 1024, 'GROUPS_PER_PROG': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 512, 'GROUPS_PER_PROG': 16}, num_warps=8, num_stages=2),
    ],
    key=['IN_FEATURES', 'OUT_FEATURES'],
)
@triton.jit
def stage1_kernel(
    x_ptr, w_ptr, b_ptr, pooled_ptr,
    BATCH, IN_FEATURES, OUT_FEATURES, N_GROUPS,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUPS_PER_PROG: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)  # block of groups index

    # output rows handled: GROUPS_PER_PROG * KERNEL_SIZE
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
        x_vals = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0)  # [BLOCK_K]

        w_ptrs = w_ptr + offs_out[:, None] * IN_FEATURES + offs_k[None, :]
        w_mask = out_mask[:, None] & mask_k[None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [OUT_PER_PROG, BLOCK_K]

        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

    b_vals = tl.load(b_ptr + offs_out, mask=out_mask, other=0.0)
    acc = acc + b_vals

    # reshape acc into [GROUPS_PER_PROG, KERNEL_SIZE] and max over KERNEL_SIZE
    acc2d = tl.reshape(acc, (GROUPS_PER_PROG, KERNEL_SIZE))
    pooled = tl.max(acc2d, axis=1)  # [GROUPS_PER_PROG]

    # write pooled[pid_b, base_group : base_group + GROUPS_PER_PROG]
    base_group = pid_g * GROUPS_PER_PROG
    offs_g = base_group + tl.arange(0, GROUPS_PER_PROG)
    g_mask = offs_g < N_GROUPS
    tl.store(pooled_ptr + pid_b * N_GROUPS + offs_g, pooled, mask=g_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
    ],
    key=['N_GROUPS'],
)
@triton.jit
def stage2_kernel(
    pooled_ptr, out_ptr,
    BATCH, N_GROUPS,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    row_ptr = pooled_ptr + pid_b * N_GROUPS

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k_start in range(0, N_GROUPS, BLOCK):
        offs = k_start + tl.arange(0, BLOCK)
        mask = offs < N_GROUPS
        v = tl.load(row_ptr + offs, mask=mask, other=0.0)
        acc += v

    s = tl.sum(acc, axis=0) * SCALE
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

        pooled = torch.empty((B, n_groups), device=device, dtype=torch.float32)
        out = torch.empty(B, device=device, dtype=torch.float32)

        def grid1(meta):
            gpp = meta['GROUPS_PER_PROG']
            return (B, triton.cdiv(n_groups, gpp))

        stage1_kernel[grid1](
            x, W, b, pooled,
            B, self.in_features, self.out_features, n_groups,
            KERNEL_SIZE=self.kernel_size,
        )

        stage2_kernel[(B,)](
            pooled, out,
            B, n_groups,
            SCALE=self.scale_factor,
        )
        return out