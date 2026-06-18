import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_gemm_lse_act_kernel(
    A_ptr, B_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_PID_N: tl.constexpr,
):
    """
    Persistent-per-M-tile kernel:
      - Each program owns a tile of BLOCK_M rows
      - Streams through N tiles, computing partial GEMM (in bf16 dot)
      - Performs online LSE across the N dimension on-the-fly
      - At the end, applies leakyrelu*2, gelu*2 and writes one value per row
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    # online LSE state per row in the tile
    neg_inf = float('-inf')
    running_max = tl.full([BLOCK_M], neg_inf, dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_M], dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    for pid_n in range(0, NUM_PID_N):
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        for k in range(0, K, BLOCK_K):
            k_remain = K - k
            a = tl.load(a_ptrs, mask=(m_mask[:, None]) & ((offs_k[None, :]) < k_remain), other=0.0)
            b = tl.load(b_ptrs, mask=((offs_k[:, None]) < k_remain) & (n_mask[None, :]), other=0.0)
            acc += tl.dot(a, b, allow_tf32=True)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            acc = acc + bias[None, :]

        # mask invalid n columns to -inf so they don't affect LSE
        acc = tl.where(n_mask[None, :], acc, neg_inf)

        tile_max = tl.max(acc, axis=1)  # [BLOCK_M]
        new_max = tl.maximum(running_max, tile_max)
        # rescale running_sum
        running_sum = running_sum * tl.exp(running_max - new_max)
        running_sum += tl.sum(tl.exp(acc - new_max[:, None]), axis=1)
        running_max = new_max

    lse = running_max + tl.log(running_sum)

    # LeakyReLU twice (slope 0.01)
    x = tl.where(lse >= 0.0, lse, lse * 0.01)
    x = tl.where(x >= 0.0, x, x * 0.01)

    # GELU twice (exact form with erf)
    inv_sqrt2 = 0.7071067811865475
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(out_ptr + offs_m, x, mask=m_mask)


def fused_gemm_lse_act(x_bf16, w_bf16_t, bias):
    """
    x_bf16: (M, K)  bf16, contiguous
    w_bf16_t: (K, N) bf16, contiguous (transposed weight)
    bias: (N,) fp32 or None
    Returns: (M, 1) fp32
    """
    M, K = x_bf16.shape
    K2, N = w_bf16_t.shape
    assert K == K2

    out = torch.empty((M, 1), device=x_bf16.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 32
    NUM_PID_N = (N + BLOCK_N - 1) // BLOCK_N

    grid = ((M + BLOCK_M - 1) // BLOCK_M,)

    fused_gemm_lse_act_kernel[grid](
        x_bf16, w_bf16_t,
        bias if bias is not None else x_bf16,  # dummy ptr if no bias
        out,
        M, N, K,
        x_bf16.stride(0), x_bf16.stride(1),
        w_bf16_t.stride(0), w_bf16_t.stride(1),
        HAS_BIAS=(bias is not None),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_PID_N=NUM_PID_N,
        num_warps=4,
        num_stages=3,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._weight_bf16_t = None
        self._bias_fp32 = None

    def _ensure_cached(self, device):
        if self._weight_bf16_t is None or self._weight_bf16_t.device != device:
            # Linear weight: (out, in). We want (in, out) bf16 for matmul.
            w = self.linear.weight.detach().to(device=device, dtype=torch.bfloat16)
            self._weight_bf16_t = w.t().contiguous()
            if self.linear.bias is not None:
                self._bias_fp32 = self.linear.bias.detach().to(device=device, dtype=torch.float32).contiguous()
            else:
                self._bias_fp32 = None

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        self._ensure_cached(x.device)
        x_bf16 = x.to(torch.bfloat16).contiguous()
        return fused_gemm_lse_act(x_bf16, self._weight_bf16_t, self._bias_fp32)