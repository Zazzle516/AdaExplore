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

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_partial_lse_kernel(
    X_ptr, W_ptr, B_ptr,
    max_ptr, sumexp_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_pm, stride_pn,
    SCALE: tl.constexpr,
    CMIN: tl.constexpr,
    CMAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs)
        w = tl.load(w_ptrs)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add bias
    b = tl.load(B_ptr + offs_n)
    acc = acc + b[None, :]
    # scale * 2
    acc = acc * SCALE
    # clamp
    acc = tl.minimum(tl.maximum(acc, CMIN), CMAX)

    mask_m = offs_m < M

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
        # Pre-transpose W to (K, N) layout for contiguous K-axis loads
        self._wt_cache = None

    def _get_wt(self):
        W = self.matmul.weight  # (N, K)
        if (self._wt_cache is None
                or self._wt_cache.device != W.device
                or self._wt_cache.dtype != W.dtype
                or self._wt_cache.data_ptr() == 0):
            self._wt_cache = W.t().contiguous()
        return self._wt_cache

    def forward(self, x):
        x = x.cuda().contiguous()
        Wt = self._get_wt().to(x.device)  # (K, N), contiguous
        B = self.matmul.bias.contiguous()  # (N,)

        M, K = x.shape
        N = Wt.shape[1]

        # Use a fixed BLOCK_N for partial buffer layout (autotune may pick others,
        # but ntiles_n must match the kernel's BLOCK_N choice).
        # We'll allocate based on max possible (smallest BLOCK_N=64).
        BLOCK_N_REF = 64
        ntiles_max = (N + BLOCK_N_REF - 1) // BLOCK_N_REF

        max_buf = torch.empty((M, ntiles_max), device=x.device, dtype=torch.float32)
        sumexp_buf = torch.empty((M, ntiles_max), device=x.device, dtype=torch.float32)

        scale2 = self.scale_factor * 2.0

        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        )

        gemm_partial_lse_kernel[grid](
            x, Wt, B,
            max_buf, sumexp_buf,
            M, N, K,
            x.stride(0), x.stride(1),
            Wt.stride(0), Wt.stride(1),
            max_buf.stride(0), max_buf.stride(1),
            SCALE=scale2,
            CMIN=self.clamp_min,
            CMAX=self.clamp_max,
        )

        # Determine actual BLOCK_N used by autotuner via best_config
        best_cfg = gemm_partial_lse_kernel.best_config
        actual_BN = best_cfg.kwargs['BLOCK_N']
        ntiles_n = (N + actual_BN - 1) // actual_BN

        out = torch.empty((M, 1), device=x.device, dtype=torch.float32)

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