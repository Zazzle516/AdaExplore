import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_swish_bias_gn_kernel(
    X_ptr, Bp_ptr, Gamma_ptr, Beta_ptr, Y_ptr,
    M, N, G, CPG,
    eps,
    BLOCK_CPG: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // G
    grp = pid % G

    offs = tl.arange(0, BLOCK_CPG)
    mask = offs < CPG

    base = row * N + grp * CPG
    x = tl.load(X_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    # swish + bias
    bp = tl.load(Bp_ptr + grp * CPG + offs, mask=mask, other=0.0).to(tl.float32)
    x = x * tl.sigmoid(x) + bp

    cnt = CPG.to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / cnt
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / cnt
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(Gamma_ptr + grp * CPG + offs, mask=mask, other=0.0)
    b = tl.load(Beta_ptr + grp * CPG + offs, mask=mask, other=0.0)

    y = xc * rstd * g + b
    tl.store(Y_ptr + base + offs, y, mask=mask)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups

        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        x = x.contiguous().cuda()
        M = x.shape[0]
        N = self.out_features

        # cuBLAS matmul + bias (fast path on 4090)
        y = torch.addmm(self.matmul.bias, x, self.matmul.weight.t())

        CPG = N // self.num_groups
        BLOCK_CPG = _next_pow2(CPG)

        out = torch.empty_like(y)
        grid = (M * self.num_groups,)
        fused_swish_bias_gn_kernel[grid](
            y, self.bias, self.group_norm.weight, self.group_norm.bias, out,
            M, N, self.num_groups, CPG,
            float(self.group_norm.eps),
            BLOCK_CPG=BLOCK_CPG,
            num_warps=2,
        )
        return out