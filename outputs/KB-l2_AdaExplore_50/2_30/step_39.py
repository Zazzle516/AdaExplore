import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
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

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(Bias + offs_n)
    acc += bias[None, :]

    offs_m_out = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_out = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + offs_m_out[:, None] * stride_cm + offs_n_out[None, :] * stride_cn
    mask = (offs_m_out[:, None] < M) & (offs_n_out[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def triton_linear(x, weight_t, bias):
    """
    x: (M, K)
    weight_t: (K, N)  -- pre-transposed for contiguous N access
    bias: (N,)
    """
    M, K = x.shape
    K2, N = weight_t.shape
    assert K == K2
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
def group_norm_hardtanh_kernel(
    X, Y, W, Bias,
    C, CPG,
    EPS: tl.constexpr,
    HMIN: tl.constexpr, HMAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    grp = tl.program_id(1)

    start = grp * CPG
    offs = tl.arange(0, BLOCK)
    mask = offs < CPG

    x_ptr = X + row * C + start + offs
    x = tl.load(x_ptr, mask=mask, other=0.0)

    sum_x = tl.sum(x, axis=0)
    mean = sum_x / CPG
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / CPG
    rstd = tl.rsqrt(var + EPS)

    w = tl.load(W + start + offs, mask=mask, other=0.0)
    b = tl.load(Bias + start + offs, mask=mask, other=0.0)

    y = (x - mean) * rstd * w + b
    y = tl.minimum(tl.maximum(y, HMIN), HMAX)

    y_ptr = Y + row * C + start + offs
    tl.store(y_ptr, y, mask=mask)


def triton_groupnorm_hardtanh(x, weight, bias, num_groups, eps, hmin, hmax):
    M, C = x.shape
    CPG = C // num_groups
    BLOCK = triton.next_power_of_2(CPG)
    out = torch.empty_like(x)
    grid = (M, num_groups)
    if BLOCK <= 256:
        nw = 2
    elif BLOCK <= 1024:
        nw = 4
    else:
        nw = 8
    group_norm_hardtanh_kernel[grid](
        x, out, weight, bias,
        C, CPG,
        EPS=eps, HMIN=hmin, HMAX=hmax,
        BLOCK=BLOCK,
        num_warps=nw, num_stages=2,
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
        # Pre-transpose weight once for contiguous N access in GEMM
        with torch.no_grad():
            wt = self.gemm.weight.detach().t().contiguous()
        self.register_buffer('weight_t', wt)

    def forward(self, x):
        x = x.cuda().contiguous()
        if self.weight_t.device != x.device:
            self.weight_t = self.weight_t.to(x.device)
        b = self.gemm.bias.contiguous()
        y = triton_linear(x, self.weight_t, b)

        gw = self.group_norm.weight.contiguous()
        gb = self.group_norm.bias.contiguous()
        out = triton_groupnorm_hardtanh(y, gw, gb, self.num_groups, self.eps,
                                         self.hardtanh_min, self.hardtanh_max)
        return out