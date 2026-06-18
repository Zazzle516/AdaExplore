import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_logsumexp_act_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    # online logsumexp state
    row_max = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    num_n_tiles = tl.cdiv(N, BLOCK_N)

    for n_idx in range(0, num_n_tiles):
        offs_n = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # GEMM over K
        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + offs_k
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
            # W is (K, N) contiguous
            w_ptrs = w_ptr + k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn
            x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
            w_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)
            w = tl.load(w_ptrs, mask=w_mask, other=0.0)
            acc += tl.dot(x, w)

        b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + b[None, :]
        n_mask = offs_n[None, :] < N
        acc = tl.where(n_mask, acc, -float('inf'))

        tile_max = tl.max(acc, axis=1)
        new_max = tl.maximum(row_max, tile_max)
        # rescale previous sum
        scale = tl.exp(row_max - new_max)
        row_sum = row_sum * scale
        # add this tile's contribution
        e = tl.exp(acc - new_max[:, None])
        e = tl.where(n_mask, e, 0.0)
        row_sum = row_sum + tl.sum(e, axis=1)
        row_max = new_max

    # finalize
    x = row_max + tl.log(row_sum)
    # LeakyReLU twice (slope 0.01) — applied to scalar; both have same effect since once non-negative stays non-negative
    x = tl.where(x >= 0, x, x * 0.01)
    x = tl.where(x >= 0, x, x * 0.01)
    inv_sqrt2 = 0.7071067811865475
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    m_mask = offs_m < M
    tl.store(out_ptr + offs_m, x, mask=m_mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        # Pre-transpose and cache W as (K, N) contiguous
        with torch.no_grad():
            wt = self.linear.weight.detach().t().contiguous()
        self.register_buffer('w_kn', wt)
        if bias:
            self.register_buffer('b_buf', self.linear.bias.detach().contiguous())
        else:
            self.register_buffer('b_buf', torch.zeros(out_features))

    def forward(self, x):
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()
        W = self.w_kn
        if W.device != x.device:
            W = W.to(x.device)
            self.w_kn = W
        b = self.b_buf
        if b.device != x.device:
            b = b.to(x.device)
            self.b_buf = b

        M, K = x.shape
        N = W.shape[1]

        out = torch.empty(M, device=x.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
        fused_gemm_logsumexp_act_kernel[grid](
            x, W, b, out,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
        )

        return out.view(M, 1)