import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------
# Kernel 1: GEMM + bias + sigmoid
#   Y = sigmoid(X @ W1^T + b1)
#   X: [M, K], W: [N, K] (linear weight), b: [N]
#   Out: [M, N]
# ---------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_sigmoid_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # W is pre-transposed: [K, N]
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
    acc += b[None, :]
    # sigmoid
    acc = 1.0 / (1.0 + tl.exp(-acc))

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


# ---------------------------------------------------------------
# Kernel 2: GEMM + bias + per-row LogSumExp (online)
#   For output_size = 1024 fits in BLOCK_N tile entirely
#   X: [M, K], W: [N, K], b: [N]
#   Out: [M]   (logsumexp over N per row)
# ---------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_logsumexp_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    n_tiles = tl.cdiv(N, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    running_max = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for nt in range(0, n_tiles):
        offs_n = nt * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        # W pre-transposed: [K, N]
        w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

        for k in range(0, K, BLOCK_K):
            k_remaining = K - k
            mask_k = offs_k < k_remaining
            x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc += tl.dot(x, w)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk

        b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        acc += b[None, :]
        acc = tl.where(mask_n[None, :], acc, -float('inf'))

        tile_max = tl.max(acc, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        scale_old = tl.exp(running_max - new_max)
        scale_old = tl.where(running_max == -float('inf'), 0.0, scale_old)
        tile_sum = tl.sum(tl.exp(acc - new_max[:, None]), axis=1)
        running_sum = running_sum * scale_old + tile_sum
        running_max = new_max

    lse = running_max + tl.log(running_sum)
    tl.store(out_ptr + offs_m, lse, mask=mask_m)


def gemm_sigmoid(x, w_t, b, N):
    # w_t: [K, N] contiguous (pre-transposed)
    M, K = x.shape
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_sigmoid_kernel[grid](
        x, w_t, b, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


def gemm_logsumexp(x, w_t, b, N, BLOCK_N=256):
    M, K = x.shape
    out = torch.empty((M,), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
    gemm_logsumexp_kernel[grid](
        x, w_t, b, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        BLOCK_N=BLOCK_N,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Pre-transpose weights once: linear weight is [N, K] -> we want [K, N] contiguous
        with torch.no_grad():
            w1t = self.linear1.weight.t().contiguous()
            w2t = self.linear2.weight.t().contiguous()
        self.register_buffer('w1_t', w1t)
        self.register_buffer('w2_t', w2t)
        self._w_synced = False

    def _sync_weights(self):
        # Re-sync in case weights were modified after init
        with torch.no_grad():
            self.w1_t.copy_(self.linear1.weight.t().contiguous())
            self.w2_t.copy_(self.linear2.weight.t().contiguous())
        self._w_synced = True

    def forward(self, x):
        if not self._w_synced:
            self._sync_weights()
        x = x.contiguous()
        b1 = self.linear1.bias.contiguous()
        h = gemm_sigmoid(x, self.w1_t, b1, self.hidden_size)
        b2 = self.linear2.bias.contiguous()
        out = gemm_logsumexp(h, self.w2_t, b2, self.output_size, BLOCK_N=256)
        return out