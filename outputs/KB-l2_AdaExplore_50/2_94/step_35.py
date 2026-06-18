import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_htanh_mish_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
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
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # bias
    bias = tl.load(Bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # hardtanh: clamp(-1, 1)
    acc = tl.minimum(tl.maximum(acc, -1.0), 1.0)

    # mish: x * tanh(softplus(x)) = x * tanh(log(1+exp(x)))
    sp = tl.log(1.0 + tl.exp(acc))
    # tanh via exp
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    acc = acc * th

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def groupnorm_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr,
    M, C, G, GROUP_SIZE,
    eps,
    BLOCK_GS: tl.constexpr,
):
    # one program per (row, group)
    pid = tl.program_id(0)
    row = pid // G
    grp = pid % G

    offs = tl.arange(0, BLOCK_GS)
    mask = offs < GROUP_SIZE

    base = row * C + grp * GROUP_SIZE
    x = tl.load(X_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / GROUP_SIZE
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / GROUP_SIZE
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W_ptr + grp * GROUP_SIZE + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + grp * GROUP_SIZE + offs, mask=mask, other=0.0)

    y = xc * rstd * w + b
    tl.store(Y_ptr + base + offs, y, mask=mask)


def fused_gemm_bias_htanh_mish(x, w, b_linear, b_extra):
    # x: [M,K], w: [N,K], output: x @ w.T + b_linear + b_extra
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    bias_total = b_linear + b_extra
    # B is w.T, shape [K, N]
    B = w.t()
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_bias_htanh_mish_kernel[grid](
        x, B, bias_total, out,
        M, N, K,
        x.stride(0), x.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


def groupnorm_apply(x, weight, bias, num_groups, eps=1e-5):
    M, C = x.shape
    G = num_groups
    GROUP_SIZE = C // G
    # round up to power of 2
    BLOCK_GS = triton.next_power_of_2(GROUP_SIZE)
    out = torch.empty_like(x)
    grid = (M * G,)
    groupnorm_kernel[grid](
        x, out, weight, bias,
        M, C, G, GROUP_SIZE,
        eps,
        BLOCK_GS=BLOCK_GS,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        self.num_groups = num_groups
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous()
        w = self.gemm.weight.contiguous()
        b_linear = self.gemm.bias.contiguous()
        b_extra = self.bias.contiguous()
        y = fused_gemm_bias_htanh_mish(x, w, b_linear, b_extra)
        y = groupnorm_apply(y, self.groupnorm.weight, self.groupnorm.bias,
                            self.num_groups, self.groupnorm.eps)
        return y