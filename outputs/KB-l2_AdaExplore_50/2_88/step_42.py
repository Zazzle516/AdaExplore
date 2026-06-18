import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gn_swish_mul_swish_kernel(
    X_ptr, W_ptr, B_ptr, MUL_ptr, Y_ptr,
    M, N,
    GROUP_SIZE: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    offs = tl.arange(0, BLOCK)
    mask = offs < GROUP_SIZE

    base = pid_m * N + pid_g * GROUP_SIZE
    x = tl.load(X_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
    mean = sum_x / GROUP_SIZE
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / GROUP_SIZE
    rstd = 1.0 / tl.sqrt(var + eps)

    w_offs = pid_g * GROUP_SIZE + offs
    w = tl.load(W_ptr + w_offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + w_offs, mask=mask, other=0.0).to(tl.float32)
    mul_w = tl.load(MUL_ptr + w_offs, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * w + b
    y = y * tl.sigmoid(y)
    y = y * mul_w
    y = y * tl.sigmoid(y)

    tl.store(Y_ptr + base + offs, y, mask=mask)


def triton_gn_swish_mul_swish(x, gn_w, gn_b, mul_w, num_groups, eps=1e-5):
    M, N = x.shape
    group_size = N // num_groups
    BLOCK = 1
    while BLOCK < group_size:
        BLOCK *= 2
    out = torch.empty_like(x)
    grid = (M, num_groups)
    gn_swish_mul_swish_kernel[grid](
        x, gn_w, gn_b, mul_w, out,
        M, N,
        GROUP_SIZE=group_size,
        eps=eps,
        BLOCK=BLOCK,
        num_warps=1 if BLOCK <= 64 else (2 if BLOCK <= 128 else 4),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        self.num_groups = num_groups
        self.eps = 1e-5

    def forward(self, x):
        # Use cuBLAS for the GEMM (typically faster than custom Triton at large sizes on 4090)
        y = F.linear(x, self.gemm.weight, self.gemm.bias)
        out = triton_gn_swish_mul_swish(
            y,
            self.group_norm.weight,
            self.group_norm.bias,
            self.multiply_weight,
            self.num_groups,
            self.eps,
        )
        return out