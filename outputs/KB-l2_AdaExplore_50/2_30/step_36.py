import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_bias_kernel(
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

    for k in range(0, K, BLOCK_K):
        k_remain = K - k
        a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remain), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remain) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(Bias + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_linear(x, weight, bias):
    return F.linear(x, weight, bias)


@triton.jit
def group_norm_hardtanh_kernel(
    X, Y, W, Bias,
    C, G,
    eps: tl.constexpr,
    hmin: tl.constexpr,
    hmax: tl.constexpr,
    CPG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # one program per (row, group)
    row = tl.program_id(0)
    grp = tl.program_id(1)

    start = grp * CPG
    offs = tl.arange(0, BLOCK)
    mask = offs < CPG

    base = row * C + start
    x_ptr = X + base + offs
    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)

    sum_x = tl.sum(x, axis=0)
    mean = sum_x / CPG
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / CPG
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W + start + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(Bias + start + offs, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * w + b
    y = tl.minimum(tl.maximum(y, hmin), hmax)

    y_ptr = Y + base + offs
    tl.store(y_ptr, y, mask=mask)


def triton_groupnorm_hardtanh(x, weight, bias, num_groups, eps, hmin, hmax):
    M, C = x.shape
    CPG = C // num_groups
    BLOCK = triton.next_power_of_2(CPG)
    out = torch.empty_like(x)
    grid = (M, num_groups)
    nw = 4 if BLOCK <= 1024 else 8
    group_norm_hardtanh_kernel[grid](
        x, out, weight, bias,
        C, num_groups,
        eps, hmin, hmax,
        CPG, BLOCK,
        num_warps=nw,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.num_groups = num_groups
        self.hardtanh_min = float(hardtanh_min)
        self.hardtanh_max = float(hardtanh_max)
        self.eps = 1e-5

    def forward(self, x):
        x = x.cuda().contiguous()
        w = self.gemm.weight.contiguous()
        b = self.gemm.bias.contiguous()
        y = triton_linear(x, w, b)

        gw = self.group_norm.weight.contiguous()
        gb = self.group_norm.bias.contiguous()
        out = triton_groupnorm_hardtanh(y, gw, gb, self.num_groups, self.eps,
                                         self.hardtanh_min, self.hardtanh_max)
        return out