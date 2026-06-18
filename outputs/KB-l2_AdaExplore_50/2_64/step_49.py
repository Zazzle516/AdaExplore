import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_gemm_partial_lse_kernel(
    A_ptr, B_ptr, bias_ptr,
    pmax_ptr, psum_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    N_PER_PROG: tl.constexpr,
    SPLITS: tl.constexpr,
):
    """
    2D grid: (M_tiles, SPLITS).
    Each program computes partial (max, sumexp) for its BLOCK_M rows
    over a contiguous N chunk of size N_PER_PROG.
    Writes pmax[pid_m_global*M, split], psum[..] of shape (M, SPLITS).
    """
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    n_start = pid_s * N_PER_PROG

    neg_inf = float('-inf')
    running_max = tl.full([BLOCK_M], neg_inf, dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_M], dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    # number of N tiles inside this split
    NUM_N_TILES: tl.constexpr = N_PER_PROG // BLOCK_N

    for pid_n in range(0, NUM_N_TILES):
        offs_n = n_start + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # K is assumed divisible by BLOCK_K (8192 % BLOCK_K == 0 for K-block in {32,64,128})
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=n_mask[None, :], other=0.0)
            acc += tl.dot(a, b, allow_tf32=True)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            acc = acc + bias[None, :]

        acc = tl.where(n_mask[None, :], acc, neg_inf)

        tile_max = tl.max(acc, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        running_sum = running_sum * tl.exp(running_max - new_max)
        running_sum += tl.sum(tl.exp(acc - new_max[:, None]), axis=1)
        running_max = new_max

    # Write partial max and sum to workspace [M, SPLITS]
    out_off = offs_m * SPLITS + pid_s
    tl.store(pmax_ptr + out_off, running_max, mask=m_mask)
    tl.store(psum_ptr + out_off, running_sum, mask=m_mask)


@triton.jit
def lse_combine_act_kernel(
    pmax_ptr, psum_ptr, out_ptr,
    M,
    SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    offs_s = tl.arange(0, SPLITS)
    addr = offs_m[:, None] * SPLITS + offs_s[None, :]
    pmax = tl.load(pmax_ptr + addr, mask=m_mask[:, None], other=float('-inf'))
    psum = tl.load(psum_ptr + addr, mask=m_mask[:, None], other=0.0)

    gmax = tl.max(pmax, axis=1)
    # combine sumexp scaled to global max
    scaled = psum * tl.exp(pmax - gmax[:, None])
    gsum = tl.sum(scaled, axis=1)
    lse = gmax + tl.log(gsum)

    # LeakyReLU twice
    x = tl.where(lse >= 0.0, lse, lse * 0.01)
    x = tl.where(x >= 0.0, x, x * 0.01)

    # GELU twice (exact)
    inv_sqrt2 = 0.7071067811865475
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(out_ptr + offs_m, x, mask=m_mask)


def fused_gemm_lse_act(x_bf16, w_bf16_t, bias):
    M, K = x_bf16.shape
    K2, N = w_bf16_t.shape
    assert K == K2

    out = torch.empty((M, 1), device=x_bf16.device, dtype=torch.float32)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    SPLITS = 8
    assert N % SPLITS == 0
    N_PER_PROG = N // SPLITS
    assert N_PER_PROG % BLOCK_N == 0

    pmax = torch.empty((M, SPLITS), device=x_bf16.device, dtype=torch.float32)
    psum = torch.empty((M, SPLITS), device=x_bf16.device, dtype=torch.float32)

    grid = ((M + BLOCK_M - 1) // BLOCK_M, SPLITS)

    fused_gemm_partial_lse_kernel[grid](
        x_bf16, w_bf16_t,
        bias if bias is not None else x_bf16,
        pmax, psum,
        M, N, K,
        x_bf16.stride(0), x_bf16.stride(1),
        w_bf16_t.stride(0), w_bf16_t.stride(1),
        HAS_BIAS=(bias is not None),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        N_PER_PROG=N_PER_PROG,
        SPLITS=SPLITS,
        num_warps=8,
        num_stages=3,
    )

    BLOCK_M2 = 128
    grid2 = ((M + BLOCK_M2 - 1) // BLOCK_M2,)
    lse_combine_act_kernel[grid2](
        pmax, psum, out,
        M,
        SPLITS=SPLITS,
        BLOCK_M=BLOCK_M2,
        num_warps=4,
        num_stages=2,
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