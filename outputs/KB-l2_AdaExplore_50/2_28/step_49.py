import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def norm_residual_kernel(
    X_ptr, Y_ptr, Out_ptr,
    N,
    BLOCK_N: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    row_off = row * N

    sum_x = tl.zeros((BLOCK_N,), dtype=tl.float32)
    sum_x2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(X_ptr + row_off + offs, mask=mask, other=0.0)
        sum_x += x
        sum_x2 += x * x

    s = tl.sum(sum_x, axis=0)
    s2 = tl.sum(sum_x2, axis=0)
    mean = s / N
    var = s2 / N - mean * mean
    inv = 1.0 / tl.sqrt(var + EPS)

    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(X_ptr + row_off + offs, mask=mask, other=0.0)
        y = tl.load(Y_ptr + row_off + offs, mask=mask, other=0.0)
        xn = (x - mean) * inv
        out = (xn + y) * y
        tl.store(Out_ptr + row_off + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = float(eps)
        self.momentum = momentum
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)
        # Pre-transpose weight as a buffer for fast matmul path
        self.register_buffer("Wt", self.bmm.weight.detach().t().contiguous().cuda(), persistent=False)

    def forward(self, x, y):
        W = self.bmm.weight
        b = self.bmm.bias
        # Use cuBLAS addmm for the GEMM + bias
        # If weight was updated, refresh Wt (cheap on first run)
        Wt = W.t()
        C = torch.addmm(b, x, Wt)

        M, N = C.shape
        Out = torch.empty_like(C)

        # For N=8192 on RTX 4090, use BLOCK_N=2048 with high warps
        if N <= 1024:
            BLOCK_N = 1
            p = 1
            while p < N:
                p *= 2
            BLOCK_N = p
            num_warps = 4 if BLOCK_N <= 256 else 8
        elif N <= 4096:
            BLOCK_N = 2048
            num_warps = 8
        else:
            BLOCK_N = 2048
            num_warps = 16

        norm_residual_kernel[(M,)](
            C, y, Out,
            N,
            BLOCK_N=BLOCK_N,
            EPS=self.eps,
            num_warps=num_warps,
            num_stages=2,
        )
        return Out