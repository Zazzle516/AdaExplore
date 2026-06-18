import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_partial_lse_kernel(
    A_ptr, B_ptr, bias_ptr,
    max_partial_ptr, sumexp_partial_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_pm, stride_pn,
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

    mask_m = offs_m < M

    # A: [M, K], row-major
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # B: [K, N] pre-transposed contiguous; inner-dim N is contiguous (stride_bn=1)
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n)
    acc = acc + bias[None, :]

    neg_inf = float('-inf')
    acc = tl.where(mask_m[:, None], acc, neg_inf)

    row_max = tl.max(acc, axis=1)
    shifted = acc - row_max[:, None]
    exp_vals = tl.exp(shifted)
    exp_vals = tl.where(mask_m[:, None], exp_vals, 0.0)
    row_sumexp = tl.sum(exp_vals, axis=1)

    out_m_ptr = max_partial_ptr + offs_m * stride_pm + pid_n * stride_pn
    out_s_ptr = sumexp_partial_ptr + offs_m * stride_pm + pid_n * stride_pn
    tl.store(out_m_ptr, row_max, mask=mask_m)
    tl.store(out_s_ptr, row_sumexp, mask=mask_m)


@triton.jit
def _finalize_lse_act_kernel(
    max_partial_ptr, sumexp_partial_ptr,
    out_ptr,
    M, NTILES,
    stride_pm, stride_pn,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    offs_t = tl.arange(0, BLOCK_T)
    mask_t = offs_t < NTILES

    m_ptrs = max_partial_ptr + pid * stride_pm + offs_t * stride_pn
    s_ptrs = sumexp_partial_ptr + pid * stride_pm + offs_t * stride_pn

    neg_inf = float('-inf')
    m_vals = tl.load(m_ptrs, mask=mask_t, other=neg_inf)
    s_vals = tl.load(s_ptrs, mask=mask_t, other=0.0)

    global_max = tl.max(m_vals, axis=0)
    scaled = s_vals * tl.exp(m_vals - global_max)
    scaled = tl.where(mask_t, scaled, 0.0)
    total = tl.sum(scaled, axis=0)
    lse = global_max + tl.log(total)

    x = lse
    x = tl.where(x >= 0, x, x * 0.01)
    x = tl.where(x >= 0, x, x * 0.01)

    inv_sqrt2 = 0.7071067811865475
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(out_ptr + pid, x)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def _get_weight_t(self):
        W = self.linear.weight  # [N, K]
        Wt = getattr(self, '_weight_t_cache', None)
        if (Wt is None) or (Wt.data_ptr() == 0) or (Wt.shape[0] != W.shape[1]) or (Wt.shape[1] != W.shape[0]) or (Wt.device != W.device) or (Wt.dtype != W.dtype):
            Wt = W.t().contiguous()
            self._weight_t_cache = Wt
        return Wt

    def forward(self, x):
        x = x.contiguous().cuda()
        if self.linear.weight.device != x.device:
            self.linear.to(x.device)
            self._weight_t_cache = None
        Wt = self._get_weight_t()  # [K, N] contiguous
        b = self.linear.bias
        if b is None:
            b = torch.zeros(self.out_features, device=x.device, dtype=x.dtype)
        else:
            b = b.contiguous()

        M = x.shape[0]
        K = self.in_features
        N = self.out_features

        max_ntiles = (N + 63) // 64
        max_partial = torch.empty((M, max_ntiles), device=x.device, dtype=torch.float32)
        sumexp_partial = torch.empty((M, max_ntiles), device=x.device, dtype=torch.float32)

        chosen = {}

        def grid(META):
            chosen['BLOCK_N'] = META['BLOCK_N']
            return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

        _gemm_partial_lse_kernel[grid](
            x, Wt, b,
            max_partial, sumexp_partial,
            M, N, K,
            x.stride(0), x.stride(1),
            Wt.stride(0), Wt.stride(1),
            max_partial.stride(0), max_partial.stride(1),
        )

        BLOCK_N = chosen['BLOCK_N']
        ntiles_n = (N + BLOCK_N - 1) // BLOCK_N

        out = torch.empty((M, 1), device=x.device, dtype=torch.float32)
        BLOCK_T = _next_pow2(ntiles_n)
        if BLOCK_T < 1:
            BLOCK_T = 1

        _finalize_lse_act_kernel[(M,)](
            max_partial, sumexp_partial, out,
            M, ntiles_n,
            max_partial.stride(0), max_partial.stride(1),
            BLOCK_T=BLOCK_T,
        )
        return out