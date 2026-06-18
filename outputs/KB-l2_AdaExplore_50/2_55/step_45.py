import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 512}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 1024}, num_warps=8, num_stages=2),
    ],
    key=['IN_FEATURES', 'OUT_FEATURES', 'N_SPLITS'],
)
@triton.jit
def fused_kernel(
    x_ptr, w_ptr, b_ptr, partial_ptr,
    BATCH, IN_FEATURES, OUT_FEATURES,
    N_SPLITS: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,  # OUT_FEATURES // N_SPLITS, must be multiple of 2
    KERNEL_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)  # which split of N

    # x row pointer (in fp32)
    x_row = x_ptr + pid_b * IN_FEATURES

    # base column index for this split
    n_start = pid_s * SPLIT_SIZE
    n_end = n_start + SPLIT_SIZE

    # accumulator for the pooled-and-summed value over this split
    pooled_sum = tl.zeros((), dtype=tl.float32)

    # iterate over BLOCK_N tiles within this split
    n_tile = n_start
    NUM_TILES: tl.constexpr = SPLIT_SIZE // BLOCK_N

    for tile_idx in tl.static_range(0, NUM_TILES):
        n_off = n_start + tile_idx * BLOCK_N
        offs_n = n_off + tl.arange(0, BLOCK_N)
        # accumulator [BLOCK_N] for this output tile
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k_start in range(0, IN_FEATURES, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < IN_FEATURES
            x_vals = tl.load(x_row + offs_k, mask=mask_k, other=0.0)  # [BLOCK_K]
            # weight is [OUT_FEATURES, IN_FEATURES] (fp16)
            w_ptrs = w_ptr + offs_n[:, None] * IN_FEATURES + offs_k[None, :]
            w_vals = tl.load(w_ptrs, mask=mask_k[None, :], other=0.0)
            w_f32 = w_vals.to(tl.float32)
            acc += tl.sum(w_f32 * x_vals[None, :], axis=1)

        # add bias
        b_vals = tl.load(b_ptr + offs_n)
        acc = acc + b_vals

        # max-pool of size 2: reshape [BLOCK_N/2, 2], max over axis=1
        acc2d = tl.reshape(acc, (BLOCK_N // KERNEL_SIZE, KERNEL_SIZE))
        pooled = tl.max(acc2d, axis=1)  # [BLOCK_N/2]
        pooled_sum += tl.sum(pooled, axis=0)

    # write partial sum (will be reduced across splits in stage 2)
    tl.store(partial_ptr + pid_b * N_SPLITS + pid_s, pooled_sum)


@triton.jit
def reduce_kernel(
    partial_ptr, out_ptr,
    BATCH, N_SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N_SPLITS
    v = tl.load(partial_ptr + pid_b * N_SPLITS + offs, mask=mask, other=0.0)
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

        # cache fp16 weight
        self._w_fp16 = None
        self._b_fp32 = None

    def _prep(self, device):
        if self._w_fp16 is None or self._w_fp16.device != device:
            self._w_fp16 = self.matmul.weight.detach().to(device=device, dtype=torch.float16).contiguous()
            self._b_fp32 = self.matmul.bias.detach().to(device=device, dtype=torch.float32).contiguous()

    def forward(self, x):
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()
        device = x.device
        self._prep(device)

        B = x.shape[0]
        OF = self.out_features

        # choose number of splits
        N_SPLITS = 16
        # ensure SPLIT_SIZE is multiple of kernel_size and reasonable
        while OF % N_SPLITS != 0 and N_SPLITS > 1:
            N_SPLITS //= 2
        SPLIT_SIZE = OF // N_SPLITS

        partial = torch.empty((B, N_SPLITS), device=device, dtype=torch.float32)
        out = torch.empty(B, device=device, dtype=torch.float32)

        grid1 = (B, N_SPLITS)
        fused_kernel[grid1](
            x, self._w_fp16, self._b_fp32, partial,
            B, self.in_features, self.out_features,
            N_SPLITS=N_SPLITS,
            SPLIT_SIZE=SPLIT_SIZE,
            KERNEL_SIZE=self.kernel_size,
            SCALE=self.scale_factor,
        )

        # next power of 2 >= N_SPLITS
        BLOCK = 1
        while BLOCK < N_SPLITS:
            BLOCK *= 2

        reduce_kernel[(B,)](
            partial, out,
            B, N_SPLITS=N_SPLITS,
            SCALE=self.scale_factor,
            BLOCK=BLOCK,
        )
        return out