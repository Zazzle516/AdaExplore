import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Tiled GEMM: C = A @ B^T + bias, where A is (M, K), B is (N, K) (weight), bias is (N,)
GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def linear_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_linear(x, weight, bias):
    M, K = x.shape
    N, _ = weight.shape
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


# Fused per-row instance norm + (x + y) * y
# InstanceNorm2d with shape (B, C, 1, 1) -> per (B, C) normalization over a 1-element window.
# That means normalized output is 0 for each element (variance=0, mean=x). So x_norm = 0.
# Then output = (0 + y) * y = y * y.
# BUT we still need to execute it for the safety contract. Let's actually compute it.
# Spatial dims = 1*1 = 1 element. mean = x, var = 0, normalized = (x - mean)/sqrt(var+eps) = 0.
# So result = y * y.
# Still, we run an elementwise kernel to compute (norm + y) * y where norm comes from running normalization.

@triton.jit
def fused_inorm_addmul_kernel(
    x_ptr, y_ptr, out_ptr,
    B, C,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    n_total = B * C
    mask = offs < n_total

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)

    # InstanceNorm over a single spatial element: mean=x, var=0
    # normalized = (x - x) / sqrt(0 + eps) = 0
    mean = x
    var = 0.0
    inv = 1.0 / tl.sqrt(var + eps)
    norm = (x - mean) * inv

    out = (norm + y) * y
    tl.store(out_ptr + offs, out, mask=mask)


def fused_inorm_addmul(x, y, eps):
    B, C = x.shape
    out = torch.empty_like(x)
    n = B * C
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    fused_inorm_addmul_kernel[grid](x, y, out, B, C, eps, BLOCK=BLOCK)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)
        self.eps = eps
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, y):
        x = x.contiguous()
        y = y.contiguous()
        weight = self.bmm.weight.contiguous()
        bias = self.bmm.bias.contiguous()
        x = triton_linear(x, weight, bias)
        out = fused_inorm_addmul(x, y, self.eps)
        return out