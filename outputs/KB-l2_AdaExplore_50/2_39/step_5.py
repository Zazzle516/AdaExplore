import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_scale_kernel(
    A_ptr, B_ptr, C_ptr, bias_ptr, scale_ptr,
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
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
    scale = tl.load(scale_ptr + offs_cn, mask=offs_cn < N, other=0.0)
    acc = (acc + bias[None, :]) * scale[None, :]

    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def stats_kernel(
    Y_ptr, sum_ptr, sumsq_ptr,
    M, N,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # grid: (num_tiles_n,) - one program per column tile, reduces full M deterministically
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    s = tl.zeros((BLOCK_N,), dtype=tl.float32)
    sq = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for m_start in range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        m_mask = offs_m < M
        y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        mask = m_mask[:, None] & n_mask[None, :]
        y = tl.load(y_ptrs, mask=mask, other=0.0).to(tl.float32)
        s += tl.sum(y, axis=0)
        sq += tl.sum(y * y, axis=0)

    tl.store(sum_ptr + offs_n, s, mask=n_mask)
    tl.store(sumsq_ptr + offs_n, sq, mask=n_mask)


@triton.jit
def bn_apply_kernel(
    Y_ptr, sum_ptr, sumsq_ptr,
    weight_ptr, bias_ptr,
    run_mean_ptr, run_var_ptr,
    M, N, eps, momentum,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    n_mask = offs_n < N
    s = tl.load(sum_ptr + offs_n, mask=n_mask, other=0.0)
    sq = tl.load(sumsq_ptr + offs_n, mask=n_mask, other=0.0)

    mean = s / M
    var = sq / M - mean * mean
    # unbiased var for running stats
    inv_std = 1.0 / tl.sqrt(var + eps)

    w = tl.load(weight_ptr + offs_n, mask=n_mask, other=0.0)
    b = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)

    scale = w * inv_std
    shift = b - mean * scale

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y = tl.load(y_ptrs, mask=mask, other=0.0)
    out = y * scale[None, :] + shift[None, :]
    tl.store(y_ptrs, out, mask=mask)


@triton.jit
def update_running_stats_kernel(
    sum_ptr, sumsq_ptr,
    run_mean_ptr, run_var_ptr,
    M, N, momentum,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    s = tl.load(sum_ptr + offs, mask=mask, other=0.0)
    sq = tl.load(sumsq_ptr + offs, mask=mask, other=0.0)
    mean = s / M
    var = sq / M - mean * mean
    # unbiased
    var_unbiased = var * (M / (M - 1))

    rm = tl.load(run_mean_ptr + offs, mask=mask, other=0.0)
    rv = tl.load(run_var_ptr + offs, mask=mask, other=0.0)
    new_rm = (1.0 - momentum) * rm + momentum * mean
    new_rv = (1.0 - momentum) * rv + momentum * var_unbiased
    tl.store(run_mean_ptr + offs, new_rm, mask=mask)
    tl.store(run_var_ptr + offs, new_rv, mask=mask)


@triton.jit
def bn_eval_kernel(
    Y_ptr, run_mean_ptr, run_var_ptr,
    weight_ptr, bias_ptr,
    M, N, eps,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    n_mask = offs_n < N
    rm = tl.load(run_mean_ptr + offs_n, mask=n_mask, other=0.0)
    rv = tl.load(run_var_ptr + offs_n, mask=n_mask, other=0.0)
    w = tl.load(weight_ptr + offs_n, mask=n_mask, other=0.0)
    b = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)

    inv_std = 1.0 / tl.sqrt(rv + eps)
    scale = w * inv_std
    shift = b - rm * scale

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y = tl.load(y_ptrs, mask=mask, other=0.0)
    out = y * scale[None, :] + shift[None, :]
    tl.store(y_ptrs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.momentum = momentum

        # Linear params
        self.gemm_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.gemm_bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.gemm_weight, a=5 ** 0.5)
        fan_in = in_features
        bound = 1 / (fan_in ** 0.5)
        nn.init.uniform_(self.gemm_bias, -bound, bound)

        # Scale
        self.scale = nn.Parameter(torch.randn(scale_shape))

        # BN params
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        x = x.contiguous().cuda()
        M, K = x.shape
        N = self.out_features

        W = self.gemm_weight  # [N, K]
        # We compute Y = x @ W^T, treating B as W^T with stride_bk = 1, stride_bn = K
        Y = torch.empty((M, N), device=x.device, dtype=x.dtype)

        grid_gemm = lambda META: (
            triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        )
        gemm_scale_kernel[grid_gemm](
            x, W, Y, self.gemm_bias, self.scale,
            M, N, K,
            x.stride(0), x.stride(1),
            1, W.stride(0),  # B = W^T: stride_bk along K is W's stride along K = 1; stride_bn along N is W.stride(0)=K
            Y.stride(0), Y.stride(1),
        )

        if self.training:
            sum_buf = torch.empty(N, device=x.device, dtype=torch.float32)
            sumsq_buf = torch.empty(N, device=x.device, dtype=torch.float32)

            BLOCK_M_S = 256
            BLOCK_N_S = 64
            grid_stats = (triton.cdiv(N, BLOCK_N_S),)
            stats_kernel[grid_stats](
                Y, sum_buf, sumsq_buf,
                M, N,
                Y.stride(0), Y.stride(1),
                BLOCK_M=BLOCK_M_S, BLOCK_N=BLOCK_N_S,
                num_warps=8,
            )

            BLOCK_M_A = 128
            BLOCK_N_A = 128
            grid_apply = (triton.cdiv(M, BLOCK_M_A), triton.cdiv(N, BLOCK_N_A))
            bn_apply_kernel[grid_apply](
                Y, sum_buf, sumsq_buf,
                self.bn_weight, self.bn_bias,
                self.running_mean, self.running_var,
                M, N, self.eps, self.momentum,
                Y.stride(0), Y.stride(1),
                BLOCK_M=BLOCK_M_A, BLOCK_N=BLOCK_N_A,
            )

            BLOCK_U = 256
            grid_u = (triton.cdiv(N, BLOCK_U),)
            update_running_stats_kernel[grid_u](
                sum_buf, sumsq_buf,
                self.running_mean, self.running_var,
                M, N, self.momentum,
                BLOCK=BLOCK_U,
            )
            self.num_batches_tracked += 1
        else:
            BLOCK_M_E = 128
            BLOCK_N_E = 128
            grid_e = (triton.cdiv(M, BLOCK_M_E), triton.cdiv(N, BLOCK_N_E))
            bn_eval_kernel[grid_e](
                Y, self.running_mean, self.running_var,
                self.bn_weight, self.bn_bias,
                M, N, self.eps,
                Y.stride(0), Y.stride(1),
                BLOCK_M=BLOCK_M_E, BLOCK_N=BLOCK_N_E,
            )

        return Y