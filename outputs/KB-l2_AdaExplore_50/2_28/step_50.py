import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=2, num_stages=3),
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
        acc += tl.dot(a, b, allow_tf32=False)
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
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # one program per row; computes mean/var across N, normalizes, adds y, mul y
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x_ptrs = X_ptr + row * N + offs
    y_ptrs = Y_ptr + row * N + offs

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    # mean
    s = tl.sum(x, axis=0)
    mean = s / N
    # var
    xm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xm * xm, axis=0) / N
    inv = 1.0 / tl.sqrt(var + EPS)
    xn = xm * inv

    y = tl.load(y_ptrs, mask=mask, other=0.0).to(tl.float32)
    out = (xn + y) * y

    tl.store(Out_ptr + row * N + offs, out, mask=mask)


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
        W = self.bmm.weight.contiguous().cuda()  # (out, in)
        b = self.bmm.bias.contiguous().cuda()    # (out,)

        M, K = x.shape
        N = W.shape[0]
        # B = W^T -> (K, N)
        Bt = W.t().contiguous()

        C = torch.empty((M, N), device=x.device, dtype=x.dtype)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
        gemm_bias_kernel[grid](
            x, Bt, C, b,
            M, N, K,
            x.stride(0), x.stride(1),
            Bt.stride(0), Bt.stride(1),
            C.stride(0), C.stride(1),
        )

        # Instance norm on (M, 1, 1, N) over the spatial dims (which are size 1) means
        # normalization across a single element -> output is 0 (since (x-mean)/sqrt(var+eps) with x==mean ==0)
        # Then out = (0 + y) * y = y * y
        # Wait - InstanceNorm2d normalizes per-channel per-batch over H*W. Here H=W=1 so single element.
        # That means normalized output = 0 for all elements.
        # But original code: x.unsqueeze(1).unsqueeze(1) on (M, N) gives (M, 1, 1, N).
        # InstanceNorm2d expects (N, C, H, W). So this becomes batch=M, C=1, H=1, W=N.
        # Channels=1, so num_features mismatch with out_features=N... but InstanceNorm2d in default
        # affine=False track_running_stats=False just normalizes per (batch, channel) over (H,W)=(1,N).
        # So it normalizes across N elements per row. That's like LayerNorm across last dim.

        Out = torch.empty_like(C)
        BLOCK_N = _next_pow2(N)
        # cap block size; if N very large we need a single block since reduction is in-kernel
        # For N=8192 BLOCK_N=8192, should fit
        num_warps = 8 if BLOCK_N >= 2048 else 4
        norm_residual_kernel[(M,)](
            C, y, Out,
            N,
            EPS=self.eps,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return Out