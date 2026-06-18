import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


NORM_CONFIGS = [
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=4),
    triton.Config({}, num_warps=16, num_stages=2),
    triton.Config({}, num_warps=16, num_stages=3),
    triton.Config({}, num_warps=32, num_stages=2),
]


@triton.autotune(configs=NORM_CONFIGS, key=['N'])
@triton.jit
def norm_residual_kernel(
    X_ptr, Bias_ptr, Y_ptr, Out_ptr,
    N,
    BLOCK_N: tl.constexpr,
    EPS: tl.constexpr,
    FULL: tl.constexpr,
):
    # one program per row; computes mean/var across N, normalizes, adds y, mul y
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)

    x_ptrs = X_ptr + row * N + offs
    y_ptrs = Y_ptr + row * N + offs

    if FULL:
        x = tl.load(x_ptrs).to(tl.float32)
        bias = tl.load(Bias_ptr + offs).to(tl.float32)
        x = x + bias
        inv_n = 1.0 / N
        s = tl.sum(x, axis=0)
        mean = s * inv_n
        xm = x - mean
        var = tl.sum(xm * xm, axis=0) * inv_n
        inv = tl.rsqrt(var + EPS)
        xn = xm * inv
        y = tl.load(y_ptrs).to(tl.float32)
        out = (xn + y) * y
        tl.store(Out_ptr + row * N + offs, out)
    else:
        mask = offs < N
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(Bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x = x + bias
        inv_n = 1.0 / N
        s = tl.sum(x, axis=0)
        mean = s * inv_n
        xm = tl.where(mask, x - mean, 0.0)
        var = tl.sum(xm * xm, axis=0) * inv_n
        inv = tl.rsqrt(var + EPS)
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
        if not x.is_cuda:
            x = x.cuda()
        if not y.is_cuda:
            y = y.cuda()
        x = x.contiguous()
        y = y.contiguous()

        # matmul without bias; fuse bias add into norm kernel epilogue
        C = torch.mm(x, self.bmm.weight.t())

        M, N = C.shape
        Out = torch.empty_like(C)
        BLOCK_N = _next_pow2(N)
        FULL = (BLOCK_N == N)
        norm_residual_kernel[(M,)](
            C, self.bmm.bias, y, Out,
            N,
            BLOCK_N=BLOCK_N,
            EPS=self.eps,
            FULL=FULL,
        )
        return Out