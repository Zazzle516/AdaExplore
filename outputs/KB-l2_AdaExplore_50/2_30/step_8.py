import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_bias_kernel(
    A_ptr, B_ptr, C_ptr, bias_ptr,
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
        k_offs = k * BLOCK_K + offs_k
        a_mask = mask_m[:, None] & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & mask_n[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def group_norm_hardtanh_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr,
    M, C, G, CG,  # CG = C // G channels per group
    eps,
    hmin, hmax,
    BLOCK: tl.constexpr,
):
    # one program per (row, group)
    pid = tl.program_id(0)
    row = pid // G
    grp = pid % G

    offs = tl.arange(0, BLOCK)
    mask = offs < CG

    base = row * C + grp * CG
    x = tl.load(X_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    # mean
    sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
    mean = sum_x / CG
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / CG
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W_ptr + grp * CG + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + grp * CG + offs, mask=mask, other=0.0)

    y = (x - mean) * rstd * w + b
    y = tl.minimum(tl.maximum(y, hmin), hmax)
    tl.store(Y_ptr + base + offs, y, mask=mask)


def triton_gemm_bias(x, w, b):
    M, K = x.shape
    N, _ = w.shape
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    gemm_bias_kernel[grid](
        x, w, out, b,
        M, N, K,
        x.stride(0), x.stride(1),
        # w is (N, K); we want b[k,n]; pass w as transposed via strides
        w.stride(1), w.stride(0),
        out.stride(0), out.stride(1),
    )
    return out


def triton_group_norm_hardtanh(x, weight, bias, num_groups, eps, hmin, hmax):
    M, C = x.shape
    CG = C // num_groups
    BLOCK = triton.next_power_of_2(CG)
    out = torch.empty_like(x)
    grid = (M * num_groups,)
    group_norm_hardtanh_kernel[grid](
        x, out, weight, bias,
        M, C, num_groups, CG,
        eps, hmin, hmax,
        BLOCK=BLOCK,
        num_warps=4 if BLOCK <= 1024 else 8,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.hardtanh_min = float(hardtanh_min)
        self.hardtanh_max = float(hardtanh_max)
        self.eps = 1e-5

        gemm = nn.Linear(in_features, out_features)
        gn = nn.GroupNorm(num_groups, out_features)

        self.weight = nn.Parameter(gemm.weight.detach().clone())
        self.bias = nn.Parameter(gemm.bias.detach().clone())
        self.gn_weight = nn.Parameter(gn.weight.detach().clone())
        self.gn_bias = nn.Parameter(gn.bias.detach().clone())

    def forward(self, x):
        x = x.cuda().contiguous()
        w = self.weight.contiguous()
        b = self.bias.contiguous()
        y = triton_gemm_bias(x, w, b)
        out = triton_group_norm_hardtanh(
            y, self.gn_weight.contiguous(), self.gn_bias.contiguous(),
            self.num_groups, self.eps, self.hardtanh_min, self.hardtanh_max
        )
        return out