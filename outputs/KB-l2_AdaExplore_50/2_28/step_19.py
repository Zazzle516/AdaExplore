import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_bias_kernel(
    A_ptr, B_ptr, C_ptr, Bias_ptr,
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

    for k in range(0, K, BLOCK_K):
        k_mask = offs_k[None, :] < (K - k)
        a = tl.load(a_ptrs, mask=mask_m[:, None] & k_mask, other=0.0)
        k_mask2 = offs_k[:, None] < (K - k)
        b = tl.load(b_ptrs, mask=k_mask2 & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(Bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def norm_residual_kernel(
    X_ptr, Y_ptr, Out_ptr,
    N,
    BLOCK_N: tl.constexpr,
    EPS: tl.constexpr,
):
    # one program per row; tiled reduction across N
    row = tl.program_id(0)
    row_off = row * N

    # Pass 1: compute sum and sum of squares with tiled reduction
    sum_x = tl.zeros((BLOCK_N,), dtype=tl.float32)
    sum_x2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(X_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        sum_x += x
        sum_x2 += x * x

    s = tl.sum(sum_x, axis=0)
    s2 = tl.sum(sum_x2, axis=0)
    mean = s / N
    var = s2 / N - mean * mean
    inv = 1.0 / tl.sqrt(var + EPS)

    # Pass 2: normalize, residual + multiply, store
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(X_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(Y_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        xn = (x - mean) * inv
        out = (xn + y) * y
        tl.store(Out_ptr + row_off + offs, out, mask=mask)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = float(eps)
        self.momentum = momentum
        # match nn.Linear init
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)

    def forward(self, x, y):
        x = x.contiguous().cuda()
        y = y.contiguous().cuda()
        W = self.bmm.weight
        b = self.bmm.bias
        if not W.is_cuda:
            W = W.cuda()
        if not b.is_cuda:
            b = b.cuda()

        M, K = x.shape
        N = W.shape[0]

        # Use cuBLAS for bit-exact match with reference: x @ W.T + b
        C = torch.addmm(b, x, W.t())

        Out = torch.empty_like(C)
        # Choose tile size for tiled reduction
        if N <= 1024:
            BLOCK_N = _next_pow2(N)
            num_warps = 4 if BLOCK_N <= 256 else 8
        elif N <= 4096:
            BLOCK_N = 1024
            num_warps = 8
        else:
            BLOCK_N = 2048
            num_warps = 8
        norm_residual_kernel[(M,)](
            C, y, Out,
            N,
            BLOCK_N=BLOCK_N,
            EPS=self.eps,
            num_warps=num_warps,
            num_stages=2,
        )
        return Out