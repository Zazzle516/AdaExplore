import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# =========================================================
# Fused Swish + bias kernel (in-place):  Y = swish(Y) + bias
# =========================================================
@triton.jit
def swish_bias_kernel(
    Y_ptr, B_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    base = pid_m * N + offs_n
    y = tl.load(Y_ptr + base, mask=mask_n, other=0.0)
    b = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    y = y * tl.sigmoid(y) + b
    tl.store(Y_ptr + base, y, mask=mask_n)


# =========================================================
# GroupNorm kernel - one program per (row, group)
# =========================================================
@triton.jit
def group_norm_kernel(
    X_ptr, Y_ptr, Gamma_ptr, Beta_ptr,
    M, C, G, CPG,
    eps,
    BLOCK_CPG: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    total_groups = M * G
    group_start = pid * ROWS_PER_PROG

    offs = tl.arange(0, BLOCK_CPG)
    mask_c = offs < CPG

    for i in tl.static_range(ROWS_PER_PROG):
        gid = group_start + i
        valid = gid < total_groups
        row = gid // G
        grp = gid % G

        base = row * C + grp * CPG
        x = tl.load(X_ptr + base + offs, mask=mask_c & valid, other=0.0).to(tl.float32)

        cnt = CPG
        mean = tl.sum(x, axis=0) / cnt
        xc = tl.where(mask_c, x - mean, 0.0)
        var = tl.sum(xc * xc, axis=0) / cnt
        rstd = 1.0 / tl.sqrt(var + eps)

        g = tl.load(Gamma_ptr + grp * CPG + offs, mask=mask_c, other=0.0)
        b = tl.load(Beta_ptr + grp * CPG + offs, mask=mask_c, other=0.0)

        y = xc * rstd * g + b
        tl.store(Y_ptr + base + offs, y, mask=mask_c & valid)


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
        M, K = x.shape
        N = self.out_features

        W = self.matmul.weight  # (N, K)
        bl = self.matmul.bias   # (N,)
        bp = self.bias           # (N,)

        # Linear via cuBLAS (uses TF32 on Ampere/Ada, matches reference)
        y = F.linear(x, W, bl)

        # Fused swish + bias (in-place on y)
        BLOCK_N = 1024
        grid_sb = (M, triton.cdiv(N, BLOCK_N))
        swish_bias_kernel[grid_sb](
            y, bp, M, N,
            BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=2,
        )

        # GroupNorm
        CPG = N // self.num_groups
        BLOCK_CPG = _next_pow2(CPG)
        out = torch.empty_like(y)
        ROWS_PER_PROG = 4
        total = M * self.num_groups
        grid_gn = ((total + ROWS_PER_PROG - 1) // ROWS_PER_PROG,)
        group_norm_kernel[grid_gn](
            y, out, self.group_norm.weight, self.group_norm.bias,
            M, N, self.num_groups, CPG,
            float(self.group_norm.eps),
            BLOCK_CPG=BLOCK_CPG,
            ROWS_PER_PROG=ROWS_PER_PROG,
            num_warps=2,
            num_stages=2,
        )
        return out