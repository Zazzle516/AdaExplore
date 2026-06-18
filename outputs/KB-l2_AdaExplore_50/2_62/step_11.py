import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_bias_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remain = K - k
        a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remain), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remain) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def gn_lrelu_double_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, G, CPG,
    eps, neg_slope,
    BLOCK_N: tl.constexpr,
    G_PER_PROG: tl.constexpr,
    CPG_C: tl.constexpr,
):
    # one program per row
    pid = tl.program_id(0)
    row_ptr_x = X_ptr + pid * N
    row_ptr_y = Y_ptr + pid * N

    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(row_ptr_x + offs, mask=mask, other=0.0).to(tl.float32)

    # reshape into (G, CPG): G_PER_PROG * CPG_C == BLOCK_N (with possible padding)
    x2 = tl.reshape(x, (G_PER_PROG, CPG_C))
    mean = tl.sum(x2, axis=1) / CPG_C
    diff = x2 - mean[:, None]
    var = tl.sum(diff * diff, axis=1) / CPG_C
    invstd = 1.0 / tl.sqrt(var + eps)

    normed = diff * invstd[:, None]
    normed_flat = tl.reshape(normed, (BLOCK_N,))

    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    out = normed_flat * w + b

    out = tl.where(out >= 0, out, out * neg_slope)
    out = out + out

    tl.store(row_ptr_y + offs, out, mask=mask)


def triton_gemm_bias(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    # weight: (N, K), use as B with stride_bk = 1, stride_bn = K  -- but we need (K, N) layout
    # actually we want A @ W.T. Treat B as weight.T effectively by swapping strides.
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    gemm_bias_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # treat weight as (K, N): stride_bk = stride along K, stride_bn = along N
        out.stride(0), out.stride(1),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope
        self.cpg = hidden_size // num_groups

        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.fc.weight.contiguous().cuda()
        b = self.fc.bias.contiguous().cuda()
        gn_w = self.gn.weight.contiguous().cuda()
        gn_b = self.gn.bias.contiguous().cuda()

        y = triton_gemm_bias(x, w, b)

        M, N = y.shape
        out = torch.empty_like(y)

        BLOCK_N = N  # 8192
        CPG_C = self.cpg
        G_PER_PROG = BLOCK_N // CPG_C

        grid = (M,)
        gn_lrelu_double_kernel[grid](
            y, gn_w, gn_b, out,
            M, N, self.num_groups, self.cpg,
            self.eps, self.negative_slope,
            BLOCK_N=BLOCK_N,
            G_PER_PROG=G_PER_PROG,
            CPG_C=CPG_C,
            num_warps=8,
            num_stages=2,
        )
        return out