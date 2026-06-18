import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_clamp_partial_lse_kernel(
    x_ptr, w_ptr, b_ptr,
    max_ptr, sum_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_maxm, stride_maxt,
    stride_summ, stride_sumt,
    SCALE2: tl.constexpr,
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

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        mask_k = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]
    acc = acc * SCALE2
    acc = tl.minimum(tl.maximum(acc, CMIN), CMAX)

    # mask out invalid N positions to -inf for reduction
    neg_inf = float('-inf')
    acc = tl.where(mask_n[None, :], acc, neg_inf)

    # per-row partial max and sum_exp(x - max)
    row_max = tl.max(acc, axis=1)  # [BLOCK_M]
    row_sum = tl.sum(tl.exp(acc - row_max[:, None]), axis=1)  # [BLOCK_M]

    # store to (M, num_n_tiles) scratch
    max_ptrs = max_ptr + offs_m * stride_maxm + pid_n * stride_maxt
    sum_ptrs = sum_ptr + offs_m * stride_summ + pid_n * stride_sumt
    tl.store(max_ptrs, row_max, mask=mask_m)
    tl.store(sum_ptrs, row_sum, mask=mask_m)


@triton.jit
def combine_lse_mish_kernel(
    max_ptr, sum_ptr, out_ptr,
    M, T,
    stride_maxm, stride_maxt,
    stride_summ, stride_sumt,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    offs_t = tl.arange(0, BLOCK_T)
    mask_t = offs_t < T

    neg_inf = float('-inf')
    maxs = tl.load(max_ptr + pid * stride_maxm + offs_t * stride_maxt, mask=mask_t, other=neg_inf)
    sums = tl.load(sum_ptr + pid * stride_summ + offs_t * stride_sumt, mask=mask_t, other=0.0)

    global_max = tl.max(maxs, axis=0)
    rescaled = sums * tl.exp(maxs - global_max)
    rescaled = tl.where(mask_t, rescaled, 0.0)
    total_sum = tl.sum(rescaled, axis=0)
    lse = global_max + tl.log(total_sum)

    # x * mish(x) = lse * lse * tanh(softplus(lse))
    sp = tl.log(1.0 + tl.exp(lse))
    e2 = tl.exp(2.0 * sp)
    tanh_sp = (e2 - 1.0) / (e2 + 1.0)
    out = lse * lse * tanh_sp

    tl.store(out_ptr + pid, out)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        # Pre-transpose weight to (K, N) contiguous along N for efficient B-loads
        self.register_buffer('weight_kt', self.matmul.weight.detach().t().contiguous().cuda())

    def forward(self, x):
        x = x.contiguous().cuda()
        bias = self.matmul.bias.contiguous()
        M, K = x.shape
        K2, N = self.weight_kt.shape
        assert K == K2

        scale2 = float(self.scale_factor * 2.0)
        cmin = float(self.clamp_min)
        cmax = float(self.clamp_max)

        # We pick a fixed BLOCK_N upper bound for scratch shape; use max possible (256).
        # But autotune may pick smaller. To keep scratch shape consistent, we compute
        # T = ceil(N / BLOCK_N) inside grid lambda. Use a tensor allocated per call.

        # Allocate scratch with maximum possible T (assume BLOCK_N >= 128)
        # Use grid-based shape: we'll allocate after autotune resolves via metadata.
        # Workaround: allocate based on min BLOCK_N=128 -> T_max = ceil(N/128)
        T_max = (N + 127) // 128
        max_scratch = torch.empty((M, T_max), device=x.device, dtype=torch.float32)
        sum_scratch = torch.empty((M, T_max), device=x.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        fused_gemm_clamp_partial_lse_kernel[grid](
            x, self.weight_kt, bias,
            max_scratch, sum_scratch,
            M, N, K,
            x.stride(0), x.stride(1),
            self.weight_kt.stride(0), self.weight_kt.stride(1),
            max_scratch.stride(0), max_scratch.stride(1),
            sum_scratch.stride(0), sum_scratch.stride(1),
            SCALE2=scale2,
            CMIN=cmin,
            CMAX=cmax,
        )

        # Determine actual T used from chosen BLOCK_N
        # We don't know BLOCK_N exactly here; but only the first T_actual entries are valid
        # and the rest must be ignored. Since we allocated T_max for BLOCK_N=128, but autotune
        # might pick BLOCK_N=256 -> fewer tiles. The extra entries are uninitialized -> garbage.
        # To handle this safely, initialize scratch to (-inf, 0) so uninitialized entries don't contribute.

        # Actually we need to re-do: zero out before. To keep correctness, let's just initialize.
        # But we already launched. We need to init BEFORE launch. Redo properly:
        # (Note: above launch path is already done; rewriting logic for safety)

        out = torch.empty((M, 1), device=x.device, dtype=torch.float32)
        # We need to know actual T. Use the launched kernel's grid second dim.
        # Simpler robust approach: just use full T_max with init values. But to remain correct,
        # we adopt the strategy of reinitializing and relaunching properly below... 
        # Instead, refactor: initialize scratch to safe defaults before launch.
        # We'll handle that via a separate cleaner forward function.
        return self._forward_clean(x, bias, scale2, cmin, cmax, M, N, K)

    def _forward_clean(self, x, bias, scale2, cmin, cmax, M, N, K):
        # Allocate scratch sized for smallest BLOCK_N to give maximum T,
        # initialize with safe defaults so unused slots don't contribute.
        T_max = (N + 127) // 128
        max_scratch = torch.full((M, T_max), float('-inf'), device=x.device, dtype=torch.float32)
        sum_scratch = torch.zeros((M, T_max), device=x.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        fused_gemm_clamp_partial_lse_kernel[grid](
            x, self.weight_kt, bias,
            max_scratch, sum_scratch,
            M, N, K,
            x.stride(0), x.stride(1),
            self.weight_kt.stride(0), self.weight_kt.stride(1),
            max_scratch.stride(0), max_scratch.stride(1),
            sum_scratch.stride(0), sum_scratch.stride(1),
            SCALE2=scale2,
            CMIN=cmin,
            CMAX=cmax,
        )

        out = torch.empty((M, 1), device=x.device, dtype=torch.float32)
        # BLOCK_T must be >= T_max
        BLOCK_T = 1
        while BLOCK_T < T_max:
            BLOCK_T *= 2
        BLOCK_T = max(BLOCK_T, 8)

        combine_lse_mish_kernel[(M,)](
            max_scratch, sum_scratch, out,
            M, T_max,
            max_scratch.stride(0), max_scratch.stride(1),
            sum_scratch.stride(0), sum_scratch.stride(1),
            BLOCK_T=BLOCK_T,
            num_warps=2,
        )
        return out