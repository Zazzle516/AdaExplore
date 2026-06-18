import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_bias_hardtanh_mish_kernel(
    A_ptr, B_ptr, C_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
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

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remain = K - k * BLOCK_K
        a_mask = mask_m[:, None] & (offs_k[None, :] < k_remain)
        b_mask = (offs_k[:, None] < k_remain) & mask_n[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # add bias (bias is fused linear bias + extra bias)
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :].to(tl.float32)

    # hardtanh in [-1, 1]
    acc = tl.minimum(tl.maximum(acc, -1.0), 1.0)

    # mish: x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(acc))
    # numerically stable tanh: 2*sigmoid(2*sp) - 1
    # sp >= 0 so -2*sp <= 0, exp is bounded in (0, 1]
    tanh_sp = 2.0 / (1.0 + tl.exp(-2.0 * sp)) - 1.0
    out = acc * tanh_sp

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, out, mask=c_mask)


@triton.jit
def group_norm_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr,
    M, C, G, CPG,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # one program per (row, group)
    pid = tl.program_id(0)
    row = pid // G
    grp = pid % G

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < CPG

    base = row * C + grp * CPG
    x_ptrs = X_ptr + base + offs
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_x = tl.sum(x, axis=0)
    mean = sum_x / CPG
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / CPG
    var = tl.maximum(var, 0.0)
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W_ptr + grp * CPG + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + grp * CPG + offs, mask=mask, other=0.0)

    y = (x - mean) * rstd * w + b
    tl.store(Y_ptr + base + offs, y, mask=mask)


def next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups

        # Match reference parameter names so state_dict loading works
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        self.eps = 1e-5
        self._wt_cache = None

    def _get_wt(self):
        # Cache the transposed weight; rebuild if weight pointer changes.
        w = self.gemm.weight
        if (self._wt_cache is None) or (self._wt_cache[0] is not w):
            wt = w.detach().t().contiguous()
            self._wt_cache = (w, wt)
        return self._wt_cache[1]

    def forward(self, x):
        x = x.contiguous().cuda()
        M, K = x.shape
        N = self.out_features

        # fused bias
        fused_bias = (self.gemm.bias + self.bias).contiguous()

        # Pre-transposed weight: shape (K, N), contiguous
        Wt = self._get_wt()
        out = torch.empty((M, N), device=x.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
        gemm_bias_hardtanh_mish_kernel[grid](
            x, Wt, out, fused_bias,
            M, N, K,
            x.stride(0), x.stride(1),
            Wt.stride(0), Wt.stride(1),
            out.stride(0), out.stride(1),
        )

        # GroupNorm
        CPG = N // self.num_groups
        BLOCK = next_pow2(CPG)
        y = torch.empty_like(out)
        grid2 = (M * self.num_groups,)
        group_norm_kernel[grid2](
            out, y, self.groupnorm.weight, self.groupnorm.bias,
            M, N, self.num_groups, CPG,
            self.eps,
            BLOCK_SIZE=BLOCK,
            num_warps=4,
        )
        return y