import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
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

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        mask_k = k_offs < K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def gn_leakyrelu_double_kernel(
    X_ptr, Y_ptr, gamma_ptr, beta_ptr,
    N, C, G,
    CPG: tl.constexpr,
    EPS: tl.constexpr,
    NEG_SLOPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # one program per (row, group)
    pid = tl.program_id(0)
    row = pid // G
    grp = pid % G

    x_row_ptr = X_ptr + row * C + grp * CPG
    y_row_ptr = Y_ptr + row * C + grp * CPG

    offs = tl.arange(0, BLOCK)
    mask = offs < CPG

    x = tl.load(x_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # mean
    sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
    inv_cpg = 1.0 / float(CPG)
    mean = sum_x * inv_cpg
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) * inv_cpg
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(gamma_ptr + grp * CPG + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(beta_ptr + grp * CPG + offs, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * g + b
    # leaky relu
    y = tl.where(y >= 0, y, y * NEG_SLOPE)
    # x + x
    y = y + y

    tl.store(y_row_ptr + offs, y, mask=mask)


def triton_linear(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    # weight: (N, K) -> B: (K, N) via transpose
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_bias_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # B = W^T
        out.stride(0), out.stride(1),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size).cuda()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps).cuda()
        self.num_groups = num_groups
        self.hidden_size = hidden_size
        self.eps = eps
        self.negative_slope = negative_slope

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.fc.weight.contiguous()
        b = self.fc.bias.contiguous()
        y = triton_linear(x, w, b)

        N, C = y.shape
        G = self.num_groups
        CPG = C // G
        out = torch.empty_like(y)

        BLOCK = triton.next_power_of_2(CPG)
        grid = (N * G,)
        gn_leakyrelu_double_kernel[grid](
            y, out,
            self.gn.weight.contiguous(), self.gn.bias.contiguous(),
            N, C, G,
            CPG=CPG,
            EPS=float(self.eps),
            NEG_SLOPE=float(self.negative_slope),
            BLOCK=BLOCK,
            num_warps=4,
            num_stages=2,
        )
        return out