import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["M", "N", "K"])
@triton.jit
def fused_linear_act_kernel(
    A_ptr, B_ptr, Bias_ptr, Add_ptr, Out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        mask_k = offs_k < k_remaining
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Bias + add_value
    bias = tl.load(Bias_ptr + offs_n, mask=mask_n, other=0.0)
    addv = tl.load(Add_ptr + offs_n, mask=mask_n, other=0.0)
    x = acc + bias[None, :] + addv[None, :]

    # Swish: x * sigmoid(x)
    sig = tl.sigmoid(x)
    x = x * sig

    # Tanh
    x = x * 2.0
    e = tl.exp(x)
    inv = 1.0 / (e + 1.0)
    x = 1.0 - 2.0 * inv

    # GELU (erf version): 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.70710678118654752440
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    # Hardtanh [-1, 1]
    x = tl.minimum(tl.maximum(x, -1.0), 1.0)

    o_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(o_ptrs, x, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.cuda().contiguous()
        weight = self.matmul.weight  # (out_features, in_features)
        bias = self.matmul.bias  # (out_features,)
        add_value = self.add_value

        M, K = x.shape
        N = self.out_features

        # B = weight.T (K, N); use transposed strides directly
        B = weight  # (N, K)
        # stride for B viewed as (K, N): bk = 1, bn = K
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

        fused_linear_act_kernel[grid](
            x, B, bias, add_value, out,
            M, N, K,
            x.stride(0), x.stride(1),
            B.stride(1), B.stride(0),  # B as (K, N): stride_bk = stride along K (=1), stride_bn = stride along N (=K)
            out.stride(0), out.stride(1),
        )
        return out