import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=16, num_stages=2),
    ],
    key=['CHUNK_SIZE'],
)
@triton.jit
def softmax_partial_kernel(
    x_ptr, partial_max_ptr, partial_sum_ptr,
    B, C, S, NUM_CHUNKS, CHUNK_SIZE,
    clamp_min: tl.constexpr, clamp_max: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_chunk = tl.program_id(1)

    row_start = pid_row * S
    chunk_start = pid_chunk * CHUNK_SIZE
    chunk_end = tl.minimum(chunk_start + CHUNK_SIZE, S)

    max_val = -float('inf')
    sum_val = 0.0
    for off in range(chunk_start, chunk_end, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < chunk_end
        x = tl.load(x_ptr + row_start + idx, mask=mask, other=-float('inf'))
        x = tl.minimum(tl.maximum(x, clamp_min), clamp_max)
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(max_val, block_max)
        e = tl.exp(x - new_max)
        e = tl.where(mask, e, 0.0)
        sum_val = sum_val * tl.exp(max_val - new_max) + tl.sum(e, axis=0)
        max_val = new_max

    tl.store(partial_max_ptr + pid_row * NUM_CHUNKS + pid_chunk, max_val)
    tl.store(partial_sum_ptr + pid_row * NUM_CHUNKS + pid_chunk, sum_val)


@triton.jit
def softmax_finalize_kernel(
    x_ptr, partial_max_ptr, partial_sum_ptr, scale_ptr, out_ptr,
    B, C, S, NUM_CHUNKS,
    clamp_min: tl.constexpr, clamp_max: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_CHUNKS_P2: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_block = tl.program_id(1)

    c = pid_row % C
    scale = tl.load(scale_ptr + c)

    # Reduce partials to global max and sum
    chunk_idx = tl.arange(0, NUM_CHUNKS_P2)
    chunk_mask = chunk_idx < NUM_CHUNKS
    pmax = tl.load(partial_max_ptr + pid_row * NUM_CHUNKS + chunk_idx,
                   mask=chunk_mask, other=-float('inf'))
    psum = tl.load(partial_sum_ptr + pid_row * NUM_CHUNKS + chunk_idx,
                   mask=chunk_mask, other=0.0)
    global_max = tl.max(pmax, axis=0)
    adjusted_sum = psum * tl.exp(pmax - global_max)
    adjusted_sum = tl.where(chunk_mask, adjusted_sum, 0.0)
    global_sum = tl.sum(adjusted_sum, axis=0)
    inv_sum = 1.0 / global_sum

    # Write output for this block
    row_start = pid_row * S
    off = pid_block * BLOCK
    idx = off + tl.arange(0, BLOCK)
    mask = idx < S
    x = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
    x = tl.minimum(tl.maximum(x, clamp_min), clamp_max)
    y = tl.exp(x - global_max) * inv_sum * scale
    tl.store(out_ptr + row_start + idx, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        b, c, d, h, w = x.shape
        S = d * h * w

        x_flat = x.contiguous().view(b, c, S)
        out = torch.empty_like(x_flat)

        scale_flat = self.scale.view(c).contiguous()

        # Choose chunk size to parallelize across SMs
        # Aim for ~8 chunks per row to give 32*64*8 = 16384 programs
        NUM_CHUNKS = 8
        CHUNK_SIZE = (S + NUM_CHUNKS - 1) // NUM_CHUNKS
        # Round up to power of 2 for finalize kernel
        NUM_CHUNKS_P2 = 1
        while NUM_CHUNKS_P2 < NUM_CHUNKS:
            NUM_CHUNKS_P2 *= 2

        partial_max = torch.empty((b * c, NUM_CHUNKS), device=x.device, dtype=torch.float32)
        partial_sum = torch.empty((b * c, NUM_CHUNKS), device=x.device, dtype=torch.float32)

        grid_partial = (b * c, NUM_CHUNKS)
        softmax_partial_kernel[grid_partial](
            x_flat, partial_max, partial_sum,
            b, c, S, NUM_CHUNKS, CHUNK_SIZE,
            self.clamp_min, self.clamp_max,
        )

        FINAL_BLOCK = 4096
        num_blocks = (S + FINAL_BLOCK - 1) // FINAL_BLOCK
        grid_final = (b * c, num_blocks)
        softmax_finalize_kernel[grid_final](
            x_flat, partial_max, partial_sum, scale_flat, out,
            b, c, S, NUM_CHUNKS,
            self.clamp_min, self.clamp_max,
            BLOCK=FINAL_BLOCK,
            NUM_CHUNKS_P2=NUM_CHUNKS_P2,
            num_warps=8,
        )

        return out.view(b, c, d, h, w)