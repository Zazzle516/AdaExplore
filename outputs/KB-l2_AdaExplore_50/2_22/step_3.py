import torch
import torch.nn as nn
import triton
import triton.language as tl


# Fused GEMM (x @ W^T + b) * (2 * scale), then clamp, then row-wise logsumexp,
# then mish-combined output: y = lse * lse * sigmoid(softplus(lse)) ... wait
# Actually: x = x * mish(x) where x is the lse scalar. mish(x) = x*tanh(softplus(x))
# So output = lse * mish(lse) = lse * lse * tanh(softplus(lse))

# Strategy: one kernel computes per-row (batch) the GEMM tile producing the
# full hidden vector for that row, performing logsumexp on the fly. But hidden
# is large (8192). We do this in two kernels:
#   Kernel 1: Tiled GEMM that writes intermediate clamped values? Too much memory.
# Better: Fuse into a single kernel where each program handles one row of the
# batch, computing the full row's hidden values in chunks, applying scale*2 and
# clamp, accumulating logsumexp online, then applying mish at the end.
# But each row requires a (input_size) x (hidden_size) reduction = 8192 * 8192 ops.
# We can split: assign multiple programs per row, each handles a chunk of N (hidden),
# does partial logsumexp, then second kernel reduces.

# Approach: one program per (row, N-tile). Each computes the full K reduction
# for its N-tile to get post-linear values, applies scale*2, clamp, and computes
# partial (max, sumexp) for that tile. Write partials to a scratch buffer.
# Second kernel reduces partials per row -> lse, then applies mish-output.

@triton.jit
def gemm_partial_lse_kernel(
    X_ptr, W_ptr, B_ptr,
    max_ptr, sumexp_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_pm, stride_pn,
    SCALE: tl.constexpr,
    CMIN: tl.constexpr,
    CMAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        k_mask = k_offs < K
        x = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_n[:, None] & k_mask[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w))
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add bias
    b = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]
    # scale * 2
    acc = acc * SCALE
    # clamp
    acc = tl.minimum(tl.maximum(acc, CMIN), CMAX)

    # mask out-of-range columns to -inf for max reduction
    neg_big = tl.full((BLOCK_M, BLOCK_N), -1e30, dtype=tl.float32)
    acc = tl.where(mask_n[None, :], acc, neg_big)

    # partial max and sumexp per row in this tile
    tile_max = tl.max(acc, axis=1)
    tile_sumexp = tl.sum(tl.exp(acc - tile_max[:, None]), axis=1)

    out_off = offs_m * stride_pm + pid_n * stride_pn
    tl.store(max_ptr + out_off, tile_max, mask=mask_m)
    tl.store(sumexp_ptr + out_off, tile_sumexp, mask=mask_m)


@triton.jit
def reduce_lse_mish_kernel(
    max_ptr, sumexp_ptr, out_ptr,
    M, NTILES,
    stride_pm, stride_pn,
    BLOCK_T: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    offs_t = tl.arange(0, BLOCK_T)
    mask_t = offs_t < NTILES

    base = pid_m * stride_pm + offs_t * stride_pn
    tmax = tl.load(max_ptr + base, mask=mask_t, other=-1e30)
    tsum = tl.load(sumexp_ptr + base, mask=mask_t, other=0.0)

    global_max = tl.max(tmax, axis=0)
    adjusted = tsum * tl.exp(tmax - global_max)
    adjusted = tl.where(mask_t, adjusted, 0.0)
    total = tl.sum(adjusted, axis=0)
    lse = global_max + tl.log(total)

    # mish(lse) = lse * tanh(softplus(lse))
    # softplus(x) = log(1+exp(x))
    sp = tl.log(1.0 + tl.exp(lse))
    # tanh via exp
    e2 = tl.exp(2.0 * sp)
    tanh_sp = (e2 - 1.0) / (e2 + 1.0)
    mish_lse = lse * tanh_sp
    out_val = lse * mish_lse

    tl.store(out_ptr + pid_m, out_val)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super().__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = float(scale_factor)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        x = x.cuda().contiguous()
        W = self.matmul.weight.contiguous()  # (N, K)
        B = self.matmul.bias.contiguous()    # (N,)

        M, K = x.shape
        N = W.shape[0]

        BLOCK_M = 32
        BLOCK_N = 128
        BLOCK_K = 32

        ntiles_n = (N + BLOCK_N - 1) // BLOCK_N

        max_buf = torch.empty((M, ntiles_n), device=x.device, dtype=torch.float32)
        sumexp_buf = torch.empty((M, ntiles_n), device=x.device, dtype=torch.float32)

        grid1 = ((M + BLOCK_M - 1) // BLOCK_M, ntiles_n)
        scale2 = self.scale_factor * 2.0

        gemm_partial_lse_kernel[grid1](
            x, W, B,
            max_buf, sumexp_buf,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            max_buf.stride(0), max_buf.stride(1),
            SCALE=scale2,
            CMIN=self.clamp_min,
            CMAX=self.clamp_max,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=2,
        )

        out = torch.empty((M, 1), device=x.device, dtype=torch.float32)

        # find next power of 2 >= ntiles_n
        BLOCK_T = 1
        while BLOCK_T < ntiles_n:
            BLOCK_T *= 2
        BLOCK_T = max(BLOCK_T, 16)

        reduce_lse_mish_kernel[(M,)](
            max_buf, sumexp_buf, out,
            M, ntiles_n,
            max_buf.stride(0), max_buf.stride(1),
            BLOCK_T=BLOCK_T,
            num_warps=2,
        )

        return out