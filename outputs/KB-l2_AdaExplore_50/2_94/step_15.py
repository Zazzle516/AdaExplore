import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_act_kernel(
    A_ptr, B_ptr, BIAS_ptr, C_ptr,
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

    # A: (M, K) row-major
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # B is (K, N) contiguous (pre-transposed), inner axis N
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        mask_k = offs_k < k_remaining
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(BIAS_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # hardtanh
    acc = tl.minimum(tl.maximum(acc, -1.0), 1.0)

    # mish: x * tanh(softplus(x))
    sp = tl.log(1.0 + tl.exp(acc))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    acc = acc * th

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def gemm_bias_act(x, Wt, bias):
    M, K = x.shape
    N = Wt.shape[1]
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_bias_act_kernel[grid](
        x, Wt, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        Wt.stride(0), Wt.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


@triton.jit
def group_norm_rg_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    C, CG,
    eps,
    BLOCK_CG: tl.constexpr,
):
    # one program per (row, group)
    row = tl.program_id(0)
    g = tl.program_id(1)
    gbase = row * C + g * CG

    offs_cg = tl.arange(0, BLOCK_CG)
    mask_cg = offs_cg < CG
    n = CG.to(tl.float32)

    x = tl.load(X_ptr + gbase + offs_cg, mask=mask_cg, other=0.0)
    mean = tl.sum(x, axis=0) / n
    xc = tl.where(mask_cg, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / n
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W_ptr + g * CG + offs_cg, mask=mask_cg, other=0.0)
    b = tl.load(B_ptr + g * CG + offs_cg, mask=mask_cg, other=0.0)

    y = xc * rstd * w + b
    tl.store(Y_ptr + gbase + offs_cg, y, mask=mask_cg)


def group_norm(x, weight, bias, num_groups, eps):
    M, C = x.shape
    G = num_groups
    CG = C // G
    BLOCK_CG = triton.next_power_of_2(CG)
    out = torch.empty_like(x)
    grid = (M, G)
    group_norm_rg_kernel[grid](
        x, weight, bias, out,
        C, CG, eps,
        BLOCK_CG=BLOCK_CG,
        num_warps=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        self.num_groups = num_groups
        self.eps = 1e-5
        self._cached_wt = None
        self._cached_bias = None

    def _get_cached(self):
        # Cache transposed weight (K, N) contiguous and fused bias
        w = self.gemm.weight
        if (self._cached_wt is None or
            self._cached_wt.device != w.device or
            self._cached_wt.shape[0] != w.shape[1]):
            self._cached_wt = w.detach().t().contiguous().cuda()
            self._cached_bias = (self.gemm.bias.detach() + self.bias.detach()).contiguous().cuda()
        return self._cached_wt, self._cached_bias

    def forward(self, x):
        x = x.cuda().contiguous()
        Wt, bias = self._get_cached()
        y = gemm_bias_act(x, Wt, bias)
        out = group_norm(y, self.groupnorm.weight, self.groupnorm.bias,
                         self.num_groups, self.eps)
        return out