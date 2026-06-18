import torch
import torch.nn as nn
import triton
import triton.language as tl


# Fully fused kernel: GEMM(x @ W^T + b) * (2*scale), clamp, row-wise logsumexp,
# then output = lse * mish(lse) = lse * lse * tanh(softplus(lse)).
# Each program processes BLOCK_M rows, loops over all N-tiles, and maintains
# (running_max, running_sumexp) per row in registers. No scratch buffer.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_lse_mish_kernel(
    X_ptr, W_ptr, B_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    SCALE: tl.constexpr,
    CMIN: tl.constexpr,
    CMAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n_base = tl.arange(0, BLOCK_N)

    row_max = tl.full((BLOCK_M,), -1e30, dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    num_n_tiles = tl.cdiv(N, BLOCK_N)
    for nt in range(0, num_n_tiles):
        offs_n = nt * BLOCK_N + offs_n_base

        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            x = tl.load(x_ptrs)
            w = tl.load(w_ptrs)
            acc += tl.dot(x, w)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk

        b = tl.load(B_ptr + offs_n)
        acc = (acc + b[None, :]) * SCALE
        acc = tl.minimum(tl.maximum(acc, CMIN), CMAX)

        tile_max = tl.max(acc, axis=1)
        new_max = tl.maximum(row_max, tile_max)
        row_sum = row_sum * tl.exp(row_max - new_max) + tl.sum(tl.exp(acc - new_max[:, None]), axis=1)
        row_max = new_max

    lse = row_max + tl.log(row_sum)
    # mish(lse) = lse * tanh(softplus(lse)); softplus(x) = log1p(exp(x))
    sp = tl.log(1.0 + tl.exp(lse))
    e2 = tl.exp(2.0 * sp)
    tanh_sp = (e2 - 1.0) / (e2 + 1.0)
    out_val = lse * lse * tanh_sp

    mask_m = offs_m < M
    tl.store(out_ptr + offs_m, out_val, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super().__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = float(scale_factor)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Precompute scale*2 to fold the residual add (x + x) into the GEMM output
        self._scale2 = self.scale_factor * 2.0
        # Cache the (K, N) contiguous transpose of the weight so K is the inner
        # axis for coalesced loads in the GEMM K-loop.
        self._wt_cache = None
        self._wt_version = -1

    def _get_wt(self, device):
        W = self.matmul.weight  # (N, K)
        # Rebuild cache if device changed, dtype changed, or weight was updated.
        need_rebuild = (
            self._wt_cache is None
            or self._wt_cache.device != device
            or self._wt_cache.dtype != W.dtype
            or self._wt_version != W._version
        )
        if need_rebuild:
            self._wt_cache = W.detach().to(device).t().contiguous()
            self._wt_version = W._version
        return self._wt_cache

    def forward(self, x):
        x = x.cuda().contiguous()
        Wt = self._get_wt(x.device)  # (K, N), contiguous
        B = self.matmul.bias.contiguous()  # (N,)

        M, K = x.shape
        N = Wt.shape[1]

        out = torch.empty((M, 1), device=x.device, dtype=torch.float32)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),)

        fused_gemm_lse_mish_kernel[grid](
            x, Wt, B, out,
            M, N, K,
            x.stride(0), x.stride(1),
            Wt.stride(0), Wt.stride(1),
            SCALE=self._scale2,
            CMIN=self.clamp_min,
            CMAX=self.clamp_max,
        )

        return out