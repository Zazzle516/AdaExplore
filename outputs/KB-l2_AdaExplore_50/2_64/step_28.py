import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    # Online LSE: track running max and sumexp
    running_max = neg_inf
    running_sum = 0.0

    for n_start in range(0, N, BLOCK_N):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        vals = tl.load(row_ptr + offs * stride_n, mask=mask, other=neg_inf)
        tile_max = tl.max(vals, axis=0)
        new_max = tl.maximum(running_max, tile_max)
        # rescale running_sum
        running_sum = running_sum * tl.exp(running_max - new_max)
        # add this tile's contribution
        running_sum += tl.sum(tl.exp(vals - new_max), axis=0)
        running_max = new_max

    lse = running_max + tl.log(running_sum)

    # LeakyReLU twice (slope 0.01)
    x = tl.where(lse >= 0.0, lse, lse * 0.01)
    x = tl.where(x >= 0.0, x, x * 0.01)

    # GELU twice (exact form using erf)
    inv_sqrt2 = 0.7071067811865475
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(out_ptr + pid, x)


def fused_lse_act(gemm_out):
    M, N = gemm_out.shape
    out = torch.empty((M, 1), device=gemm_out.device, dtype=gemm_out.dtype)
    BLOCK_N = 1024
    lse_act_kernel[(M,)](
        gemm_out, out,
        M, N,
        gemm_out.stride(0), gemm_out.stride(1),
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        x = x.cuda()
        # GEMM via cuBLAS (highly optimized for fp32 on Ada)
        gemm_out = F.linear(x, self.linear.weight, self.linear.bias)
        return fused_lse_act(gemm_out)