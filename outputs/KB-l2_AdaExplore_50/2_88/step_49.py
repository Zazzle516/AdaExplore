import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
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

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n)
    acc += bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


def triton_linear(x, weight_t, bias):
    M, K = x.shape
    K2, N = weight_t.shape
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_bias_kernel[grid](
        x, weight_t, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


@triton.jit
def fused_gn_swish_mul_swish_kernel(
    X_ptr, GAMMA_ptr, BETA_ptr, MW_ptr, OUT_ptr,
    N, C, CPG: tl.constexpr, GROUPS_PER_BLOCK: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # one program per (row-block, group-block)
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    # rows handled by this program
    row_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    row_mask = row_offs < N

    # channels handled: GROUPS_PER_BLOCK groups, each CPG channels
    BLOCK_C: tl.constexpr = GROUPS_PER_BLOCK * CPG
    ch_in_block = tl.arange(0, BLOCK_C)  # [BLOCK_C]
    group_in_block = ch_in_block // CPG  # which group within this block

    c_idx = pid_g * BLOCK_C + ch_in_block  # global channel idx

    # load X: shape [BLOCK_N, BLOCK_C]
    x_ptrs = X_ptr + row_offs[:, None] * C + c_idx[None, :]
    x = tl.load(x_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)

    # Compute per-group mean/var. We have GROUPS_PER_BLOCK groups per row.
    # Reshape conceptually: [BLOCK_N, GROUPS_PER_BLOCK, CPG]
    # Use tl.where with group masks to reduce over CPG per group.
    # Easier: reshape x to (BLOCK_N, GROUPS_PER_BLOCK, CPG)
    x_r = tl.reshape(x, (BLOCK_N, GROUPS_PER_BLOCK, CPG))
    sum_x = tl.sum(x_r, axis=2)  # [BLOCK_N, GROUPS_PER_BLOCK]
    mean = sum_x / CPG
    diff = x_r - mean[:, :, None]
    var = tl.sum(diff * diff, axis=2) / CPG  # [BLOCK_N, GROUPS_PER_BLOCK]
    rstd = tl.rsqrt(var + eps)

    # broadcast mean/rstd back to [BLOCK_N, BLOCK_C]
    mean_b = tl.reshape(mean[:, :, None] + tl.zeros((1, 1, CPG), dtype=tl.float32),
                        (BLOCK_N, BLOCK_C))
    rstd_b = tl.reshape(rstd[:, :, None] + tl.zeros((1, 1, CPG), dtype=tl.float32),
                        (BLOCK_N, BLOCK_C))

    gamma = tl.load(GAMMA_ptr + c_idx).to(tl.float32)
    beta = tl.load(BETA_ptr + c_idx).to(tl.float32)
    mw = tl.load(MW_ptr + c_idx).to(tl.float32)

    y = (x - mean_b) * rstd_b * gamma[None, :] + beta[None, :]
    s1 = tl.sigmoid(y)
    y1 = y * s1
    y2 = y1 * mw[None, :]
    s2 = tl.sigmoid(y2)
    out = y2 * s2

    tl.store(OUT_ptr + row_offs[:, None] * C + c_idx[None, :], out,
             mask=row_mask[:, None])


def fused_gn_swish_mul_swish(x, gamma, beta, mw, num_groups, eps=1e-5):
    N, C = x.shape
    G = num_groups
    CPG = C // G
    out = torch.empty_like(x)

    # Choose GROUPS_PER_BLOCK and BLOCK_N
    # CPG = 32 here. We want BLOCK_C = GROUPS_PER_BLOCK * CPG to be reasonable.
    if CPG <= 8:
        GROUPS_PER_BLOCK = 16
    elif CPG <= 16:
        GROUPS_PER_BLOCK = 8
    elif CPG <= 32:
        GROUPS_PER_BLOCK = 4
    elif CPG <= 64:
        GROUPS_PER_BLOCK = 2
    else:
        GROUPS_PER_BLOCK = 1

    # Ensure divides G
    while G % GROUPS_PER_BLOCK != 0:
        GROUPS_PER_BLOCK //= 2
    GROUPS_PER_BLOCK = max(GROUPS_PER_BLOCK, 1)

    BLOCK_N = 8
    while N % BLOCK_N != 0 and BLOCK_N > 1:
        BLOCK_N //= 2

    num_group_blocks = G // GROUPS_PER_BLOCK
    grid = (triton.cdiv(N, BLOCK_N), num_group_blocks)

    BLOCK_C = GROUPS_PER_BLOCK * CPG
    if BLOCK_C * BLOCK_N >= 2048:
        num_warps = 8
    elif BLOCK_C * BLOCK_N >= 512:
        num_warps = 4
    else:
        num_warps = 2

    fused_gn_swish_mul_swish_kernel[grid](
        x, gamma, beta, mw, out,
        N, C, CPG, GROUPS_PER_BLOCK,
        eps,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        self.num_groups = num_groups
        self.eps = 1e-5
        self._cached_wt = None

    def _get_weight_t(self):
        if self._cached_wt is None or self._cached_wt.device != self.gemm.weight.device:
            self._cached_wt = self.gemm.weight.t().contiguous()
        return self._cached_wt

    def forward(self, x):
        x = x.contiguous().cuda()
        wt = self._get_weight_t()
        y = triton_linear(x, wt, self.gemm.bias)
        out = fused_gn_swish_mul_swish(
            y,
            self.group_norm.weight,
            self.group_norm.bias,
            self.multiply_weight,
            self.num_groups,
            self.eps,
        )
        return out