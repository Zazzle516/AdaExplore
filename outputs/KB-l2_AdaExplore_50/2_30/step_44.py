import torch
import torch.nn as nn
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_tf32_kernel(
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
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


@triton.jit
def group_norm_hardtanh_kernel(
    X_ptr, gamma_ptr, beta_ptr, Out_ptr,
    M, C, G,
    eps,
    hmin, hmax,
    stride_xm, stride_xc,
    C_PER_G: tl.constexpr,
    BLOCK_CG: tl.constexpr,
    HMIN: tl.constexpr,
    HMAX: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    offs = tl.arange(0, BLOCK_CG)
    c_start = pid_g * C_PER_G
    x_ptrs = X_ptr + pid_m * stride_xm + (c_start + offs) * stride_xc
    x = tl.load(x_ptrs)

    sum_x = tl.sum(x, axis=0)
    mean = sum_x / C_PER_G
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / C_PER_G
    rstd = tl.rsqrt(var + eps)

    g = tl.load(gamma_ptr + c_start + offs)
    b = tl.load(beta_ptr + c_start + offs)

    y = diff * rstd * g + b
    y = tl.minimum(tl.maximum(y, HMIN), HMAX)

    out_ptrs = Out_ptr + pid_m * stride_xm + (c_start + offs) * stride_xc
    tl.store(out_ptrs, y)


def triton_gemm(x, Bt, bias):
    M, K = x.shape
    N = Bt.shape[1]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_tf32_kernel[grid](
        x, Bt, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        Bt.stride(0), Bt.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


def triton_gn_hardtanh(x, gamma, beta, num_groups, eps, hmin, hmax):
    M, C = x.shape
    C_per_G = C // num_groups
    BLOCK_CG = triton.next_power_of_2(C_per_G)
    out = torch.empty_like(x)
    grid = (M, num_groups)
    nw = 8 if BLOCK_CG >= 512 else 4
    group_norm_hardtanh_kernel[grid](
        x, gamma, beta, out,
        M, C, num_groups,
        eps, hmin, hmax,
        x.stride(0), x.stride(1),
        C_PER_G=C_per_G,
        BLOCK_CG=BLOCK_CG,
        HMIN=hmin,
        HMAX=hmax,
        num_warps=nw,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.hardtanh_min = float(hardtanh_min)
        self.hardtanh_max = float(hardtanh_max)
        self.num_groups = num_groups
        self._wt_cache = None

    def _get_wt(self):
        w = self.gemm.weight
        if self._wt_cache is None or self._wt_cache.device != w.device:
            self._wt_cache = w.t().contiguous()
        return self._wt_cache

    def forward(self, x):
        x = x.contiguous()
        Bt = self._get_wt()
        y = triton_gemm(x, Bt, self.gemm.bias)
        out = triton_gn_hardtanh(
            y,
            self.group_norm.weight, self.group_norm.bias,
            self.num_groups, float(self.group_norm.eps),
            self.hardtanh_min, self.hardtanh_max,
        )
        return out