import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
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

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_gemm_bias(x, w_t, bias):
    M, K = x.shape
    K2, N = w_t.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    gemm_bias_kernel[grid](
        x, w_t, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


@triton.jit
def fused_gn_swish_mul_swish_kernel(
    x_ptr, gn_w_ptr, gn_b_ptr, mul_w_ptr, out_ptr,
    N, C, G, CPG,
    eps,
    BLOCK_CPG: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
):
    # one program = one (sample, group-tile) but loop over groups for amortization
    pid = tl.program_id(0)
    n = pid // tl.cdiv(G, GROUPS_PER_BLOCK)
    g_tile = pid % tl.cdiv(G, GROUPS_PER_BLOCK)

    g_start = g_tile * GROUPS_PER_BLOCK

    for g_off in tl.static_range(0, GROUPS_PER_BLOCK):
        g = g_start + g_off
        # if g < G, process
        valid_g = g < G
        c_offs = tl.arange(0, BLOCK_CPG)
        mask = (c_offs < CPG) & valid_g
        ptr_off = n * C + g * CPG + c_offs

        x = tl.load(x_ptr + ptr_off, mask=mask, other=0.0).to(tl.float32)

        # mean / var
        sum_x = tl.sum(x, axis=0)
        mean = sum_x / CPG
        diff = x - mean
        diff = tl.where(mask, diff, 0.0)
        var = tl.sum(diff * diff, axis=0) / CPG
        rstd = 1.0 / tl.sqrt(var + eps)

        gn_w = tl.load(gn_w_ptr + g * CPG + c_offs, mask=mask, other=0.0).to(tl.float32)
        gn_b = tl.load(gn_b_ptr + g * CPG + c_offs, mask=mask, other=0.0).to(tl.float32)
        mul_w = tl.load(mul_w_ptr + g * CPG + c_offs, mask=mask, other=0.0).to(tl.float32)

        normed = (x - mean) * rstd
        y = normed * gn_w + gn_b
        # swish
        y = y * tl.sigmoid(y)
        # multiply
        y = y * mul_w
        # swish again
        y = y * tl.sigmoid(y)

        tl.store(out_ptr + ptr_off, y, mask=mask)


def fused_gn_swish_mul_swish(x, gn_w, gn_b, mul_w, num_groups, eps=1e-5):
    N, C = x.shape
    G = num_groups
    CPG = C // G
    out = torch.empty_like(x)

    # Choose BLOCK_CPG = next pow2 >= CPG
    BLOCK_CPG = triton.next_power_of_2(CPG)
    GROUPS_PER_BLOCK = 8
    if G % GROUPS_PER_BLOCK != 0:
        GROUPS_PER_BLOCK = 1

    grid = (N * triton.cdiv(G, GROUPS_PER_BLOCK),)
    fused_gn_swish_mul_swish_kernel[grid](
        x, gn_w, gn_b, mul_w, out,
        N, C, G, CPG,
        eps,
        BLOCK_CPG=BLOCK_CPG,
        GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
        num_warps=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups

        # Linear: weight (out, in), bias (out,)
        lin = nn.Linear(in_features, out_features)
        # Pre-transpose weight to (in, out) for contiguous K-major loads
        self.register_buffer('weight_t', lin.weight.detach().t().contiguous())
        self.register_buffer('bias', lin.bias.detach().contiguous())

        gn = nn.GroupNorm(num_groups, out_features)
        self.register_buffer('gn_weight', gn.weight.detach().contiguous())
        self.register_buffer('gn_bias', gn.bias.detach().contiguous())
        self.eps = 1e-5

        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))

    def forward(self, x):
        x = x.contiguous()
        y = triton_gemm_bias(x, self.weight_t, self.bias)
        out = fused_gn_swish_mul_swish(
            y, self.gn_weight, self.gn_bias, self.multiply_weight.contiguous(),
            self.num_groups, self.eps
        )
        return out