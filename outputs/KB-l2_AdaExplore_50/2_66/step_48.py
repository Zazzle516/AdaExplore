import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 1}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 1}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 1}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 1}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 1}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 1}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 1}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 1}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_bias_dropout_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    seed, dropout_p, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    APPLY_DROPOUT: tl.constexpr,
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

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    if APPLY_DROPOUT:
        # generate per-element random and apply dropout mask
        rng_offs = offs_m[:, None] * N + offs_n[None, :]
        rnd = tl.rand(seed, rng_offs)
        keep = rnd > dropout_p
        acc = tl.where(keep, acc * scale, 0.0)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def softmax_resident_kernel(
    in_ptr, out_ptr, M, N,
    stride_m, stride_n,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    row_ptr = in_ptr + row * stride_m
    out_row_ptr = out_ptr + row * stride_m

    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(row_ptr + offs * stride_n, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(out_row_ptr + offs * stride_n, y, mask=mask)


def triton_linear_dropout(x, weight_t, bias, dropout_p, training):
    M, K = x.shape
    K2, N = weight_t.shape
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    apply_dropout = training and dropout_p > 0.0
    seed = torch.randint(0, 2**31 - 1, (1,), device=x.device).item() if apply_dropout else 0
    scale = 1.0 / (1.0 - dropout_p) if dropout_p < 1.0 else 0.0
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_bias_dropout_kernel[grid](
        x, weight_t, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        out.stride(0), out.stride(1),
        seed, dropout_p, scale,
        APPLY_DROPOUT=apply_dropout,
    )
    return out


def triton_softmax(x):
    M, N = x.shape
    out = torch.empty_like(x)
    # Round up to next power of two for BLOCK_N
    BLOCK_N = triton.next_power_of_2(N)
    grid = (M,)
    softmax_resident_kernel[grid](
        x, out, M, N,
        x.stride(0), x.stride(1),
        BLOCK_N=BLOCK_N,
        num_warps=16,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)
        self.dropout_p = dropout_p
        # Pre-transpose weight: (out, in) -> (in, out) contiguous
        with torch.no_grad():
            wt = self.matmul.weight.detach().t().contiguous()
        self.register_buffer('weight_t', wt)

    def forward(self, x):
        x = x.cuda().contiguous()
        bias = self.matmul.bias.contiguous()
        x = triton_linear_dropout(x, self.weight_t, bias, self.dropout_p, self.training)
        x = triton_softmax(x)
        return x