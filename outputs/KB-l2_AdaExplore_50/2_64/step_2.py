import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def lse_act_kernel(
    in_ptr, out_ptr,
    M, N,
    stride_m, stride_n,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row_ptr = in_ptr + pid * stride_m

    neg_inf = float('-inf')
    # First pass: compute max
    offs = tl.arange(0, BLOCK_N)
    running_max = tl.full((BLOCK_N,), neg_inf, dtype=tl.float32)
    n_blocks = tl.cdiv(N, BLOCK_N)
    for i in range(0, n_blocks):
        idx = i * BLOCK_N + offs
        mask = idx < N
        vals = tl.load(row_ptr + idx * stride_n, mask=mask, other=neg_inf)
        running_max = tl.maximum(running_max, vals)
    global_max = tl.max(running_max, axis=0)

    # Second pass: compute sum of exp(x - max)
    running_sum = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i in range(0, n_blocks):
        idx = i * BLOCK_N + offs
        mask = idx < N
        vals = tl.load(row_ptr + idx * stride_n, mask=mask, other=neg_inf)
        e = tl.exp(vals - global_max)
        e = tl.where(mask, e, 0.0)
        running_sum += e
    total = tl.sum(running_sum, axis=0)
    lse = global_max + tl.log(total)

    # LeakyReLU twice (slope 0.01)
    x = tl.where(lse >= 0.0, lse, lse * 0.01)
    x = tl.where(x >= 0.0, x, x * 0.01)

    # GELU twice (exact form using erf)
    inv_sqrt2 = 0.7071067811865475
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(out_ptr + pid, x)


def fused_lse_act(z):
    # z: [M, N]
    M, N = z.shape
    z = z.contiguous()
    out = torch.empty((M, 1), device=z.device, dtype=z.dtype)
    BLOCK_N = 1024
    num_warps = 8
    lse_act_kernel[(M,)](
        z, out,
        M, N,
        z.stride(0), z.stride(1),
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        x = x.cuda()
        # Use cuBLAS for the heavy GEMM (highly optimized for 1024x8192x8192 fp32)
        z = torch.nn.functional.linear(x, self.linear.weight, self.linear.bias)
        return fused_lse_act(z)