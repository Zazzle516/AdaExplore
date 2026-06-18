import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Strategy:
# The heavy op is ConvTranspose2d. To satisfy the safety contract we must
# still perform all the conv-transpose multiply-adds, but we don't have to
# materialize the full H_out x W_out output. We implement a custom Triton
# kernel that:
#   - iterates over input (n, ic_tile, h_in, w_in)
#   - computes input * weight outer products
#   - accumulates each contribution directly into the per-(n, oc) sum
#     reduction (atomic_add into a small [N, OC] buffer)
#
# Because the output of the conv-transpose is summed over H_out, W_out by
# the global average pool, the contribution of each (input pixel, kernel
# element) to the per-channel sum is simply x[n,ic,h,w] * w[ic,oc,kh,kw]
# whenever the corresponding output spatial location is in-bounds. With
# stride=1, padding=0, and H_in << H_out (514×514), almost every output
# location is in-bounds. We still iterate kh*kw for each input pixel ->
# same multiply-add count.
#
# To avoid atomics being a bottleneck, we use a 2-stage reduction:
#   stage 1: one program per (n, ic_tile, h_in tile) writes a partial
#            [N, OC, NUM_BLOCKS] buffer (sum over its assigned input slice
#            * kernel) — accumulated via atomic_add into per-block bins.
# Simpler approach: directly accumulate into [N, OC] with atomic_add.
# OC=128, N=16 -> only 2048 distinct atomic targets — manageable.
#
# Each kernel program handles one (n, h_in tile of BLOCK_H rows, all w_in,
# subset of IC). For each (kh, kw) it computes a partial dot product of
# the input slice with weight[:,oc_tile,kh,kw] reduced over the input
# spatial slice (this reduction collapses H_in*W_in into a scalar per OC
# for that (n, kh, kw, ic_tile)). The sum-over-output-spatial of conv
# transpose is, ignoring the small boundary effects, just
#   sum_{h,w} x[n,ic,h,w] * w[ic,oc,kh,kw]
# = (sum_{h,w} x[n,ic,h,w]) * w[ic,oc,kh,kw]  for full overlap
# BUT THIS IS FORBIDDEN by the safety contract.
#
# Therefore we revert to running cuDNN's conv_transpose2d to do the heavy
# work, and we focus optimization on the post-reduction.
#
# The pool's best kernels run at ~20.5 ms, with cuDNN dominating. We can't
# beat cuDNN at conv_transpose for this size easily. Instead, optimize the
# tail aggressively and ensure we don't add any extra passes.


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=16, num_stages=2),
    ],
    key=['CHUNK'],
)
@triton.jit
def _partial_sum_kernel(
    y_ptr,           # [N, OC, HW]
    partial_ptr,     # [N, OC, SPLIT]
    HW,
    CHUNK,
    stride_n, stride_c,
    OC,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_s = tl.program_id(2)
    base = pid_n * stride_n + pid_c * stride_c
    start = pid_s * CHUNK
    end = start + CHUNK
    if end > HW:
        end = HW

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    off = start
    while off < end:
        idx = off + tl.arange(0, BLOCK)
        mask = idx < end
        v = tl.load(y_ptr + base + idx, mask=mask, other=0.0)
        acc += v
        off += BLOCK
    total = tl.sum(acc, axis=0)
    tl.store(partial_ptr + (pid_n * OC + pid_c) * SPLIT + pid_s, total)


@triton.jit
def _finalize_kernel(
    partial_ptr,     # [N, OC, SPLIT]
    bias_ptr,        # [OC]
    out_ptr,         # [N]
    OC,
    HW,
    SPLIT: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_OC)
    mask_c = offs_c < OC

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    for s in tl.static_range(0, SPLIT):
        v = tl.load(partial_ptr + (pid * OC + offs_c) * SPLIT + s, mask=mask_c, other=0.0)
        acc += v

    mean = acc / HW
    b = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
    z = mean + b
    z_masked = tl.where(mask_c, z, -float('inf'))
    mx = tl.max(z_masked, axis=0)
    e = tl.exp(z_masked - mx)
    e = tl.where(mask_c, e, 0.0)
    ssum = tl.sum(e, axis=0)
    lse = mx + tl.log(ssum)
    tl.store(out_ptr + pid, lse * 10.0)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.out_channels = out_channels

    def forward(self, x):
        y = self.conv_transpose(x)
        N, OC, H, W = y.shape
        HW = H * W

        y = y.contiguous()

        SPLIT = 32
        CHUNK = (HW + SPLIT - 1) // SPLIT

        partial = torch.empty((N, OC, SPLIT), device=y.device, dtype=torch.float32)

        grid = (N, OC, SPLIT)
        _partial_sum_kernel[grid](
            y, partial, HW, CHUNK,
            y.stride(0), y.stride(1),
            OC,
            SPLIT=SPLIT,
        )

        out = torch.empty((N,), device=y.device, dtype=torch.float32)
        bias_flat = self.bias.view(-1).contiguous()

        BLOCK_OC = 1
        while BLOCK_OC < OC:
            BLOCK_OC *= 2

        _finalize_kernel[(N,)](
            partial, bias_flat, out, OC, HW,
            SPLIT=SPLIT,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
        )

        return out.view(N, 1)