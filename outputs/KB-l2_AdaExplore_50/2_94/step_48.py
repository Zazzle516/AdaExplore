import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_fused_kernel(
    A_ptr, B_ptr, bias1_ptr, bias2_ptr, C_ptr,
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

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # bias1 (linear bias) + bias2 (extra bias)
    offs_n_full = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n_full < N
    b1 = tl.load(bias1_ptr + offs_n_full, mask=n_mask, other=0.0)
    b2 = tl.load(bias2_ptr + offs_n_full, mask=n_mask, other=0.0)
    acc = acc + b1[None, :] + b2[None, :]

    # hardtanh in [-1, 1]
    acc = tl.minimum(tl.maximum(acc, -1.0), 1.0)
    # mish: x * tanh(softplus(x)); softplus = log(1+exp(x))
    # since acc in [-1,1], exp is fine
    sp = tl.log(1.0 + tl.exp(acc))
    # tanh via exp
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    out = acc * th

    offs_m_full = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_ptrs = C_ptr + offs_m_full[:, None] * stride_cm + offs_n_full[None, :] * stride_cn
    c_mask = (offs_m_full[:, None] < M) & (offs_n_full[None, :] < N)
    tl.store(c_ptrs, out, mask=c_mask)


@triton.jit
def group_norm_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr,
    M, C, G, CPG: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    # one program per (row, group)
    pid = tl.program_id(0)
    row = pid // G
    grp = pid % G

    offs = tl.arange(0, BLOCK)
    mask = offs < CPG

    base = row * C + grp * CPG
    x = tl.load(X_ptr + base + offs, mask=mask, other=0.0)
    x_f = x.to(tl.float32)

    s = tl.sum(tl.where(mask, x_f, 0.0), axis=0)
    mean = s / CPG
    d = tl.where(mask, x_f - mean, 0.0)
    var = tl.sum(d * d, axis=0) / CPG
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W_ptr + grp * CPG + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + grp * CPG + offs, mask=mask, other=0.0)

    y = (x_f - mean) * rstd * w + b
    tl.store(Y_ptr + base + offs, y, mask=mask)


def _next_pow2(x):
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

        gemm = nn.Linear(in_features, out_features)
        # store weight transposed as (K, N) contiguous
        self.weight_t = nn.Parameter(gemm.weight.detach().t().contiguous())
        self.gemm_bias = nn.Parameter(gemm.bias.detach().contiguous())
        self.bias = nn.Parameter(torch.randn(bias_shape))

        gn = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        self.gn_weight = nn.Parameter(gn.weight.detach().contiguous())
        self.gn_bias = nn.Parameter(gn.bias.detach().contiguous())
        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous()
        M, K = x.shape
        N = self.out_features

        out = torch.empty((M, N), device=x.device, dtype=torch.float32)

        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 64
        GROUP_M = 8

        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        gemm_fused_kernel[grid](
            x, self.weight_t, self.gemm_bias, self.bias, out,
            M, N, K,
            x.stride(0), x.stride(1),
            self.weight_t.stride(0), self.weight_t.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M,
            num_warps=8, num_stages=4,
        )

        # GroupNorm
        CPG = N // self.num_groups  # 32
        BLOCK = _next_pow2(CPG)
        y = torch.empty_like(out)
        grid2 = (M * self.num_groups,)
        group_norm_kernel[grid2](
            out, y, self.gn_weight, self.gn_bias,
            M, N, self.num_groups, CPG,
            self.eps,
            BLOCK=BLOCK,
            num_warps=1, num_stages=2,
        )
        return y