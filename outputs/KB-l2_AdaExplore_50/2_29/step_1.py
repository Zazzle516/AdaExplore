import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_mish_mish_kernel(
    A, B, Bias, C,
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

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_k[None, :] < k_remaining) & mask_m[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(Bias + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # First mish: x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x)), stable: max(x,0) + log1p(exp(-|x|))
    # tanh via sigmoid: tanh(y) = 2*sigmoid(2y) - 1
    sp1 = tl.where(acc > 0, acc, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(acc)))
    tanh_sp1 = 2.0 * tl.sigmoid(2.0 * sp1) - 1.0
    x1 = acc * tanh_sp1

    sp2 = tl.where(x1 > 0, x1, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x1)))
    tanh_sp2 = 2.0 * tl.sigmoid(2.0 * sp2) - 1.0
    x2 = x1 * tanh_sp2

    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, x2, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.linear.weight.contiguous().cuda()  # (out, in)
        b = self.linear.bias.contiguous().cuda()
        M, K = x.shape
        N = self.out_features
        # We compute x @ W.T => use B = W.T
        Wt = W.t().contiguous()
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
        matmul_bias_mish_mish_kernel[grid](
            x, Wt, b, out,
            M, N, K,
            x.stride(0), x.stride(1),
            Wt.stride(0), Wt.stride(1),
            out.stride(0), out.stride(1),
        )
        return out