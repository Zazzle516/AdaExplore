import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


NORM_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=16, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=3),
    triton.Config({}, num_warps=16, num_stages=3),
]


@triton.autotune(configs=NORM_CONFIGS, key=['N'])
@triton.jit
def norm_residual_kernel(
    X_ptr, Y_ptr, Out_ptr,
    N,
    BLOCK_N: tl.constexpr,
    EPS: tl.constexpr,
):
    # one program per row; computes mean/var across N, normalizes, adds y, mul y
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x_ptrs = X_ptr + row * N + offs
    y_ptrs = Y_ptr + row * N + offs

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    # mean
    inv_n = 1.0 / N
    s = tl.sum(x, axis=0)
    mean = s * inv_n
    # var
    xm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xm * xm, axis=0) * inv_n
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
        if not x.is_cuda:
            x = x.cuda()
        if not y.is_cuda:
            y = y.cuda()
        x = x.contiguous()
        y = y.contiguous()

        # cuBLAS linear (guaranteed correctness, very fast on 4090)
        C = F.linear(x, self.bmm.weight, self.bmm.bias)

        M, N = C.shape
        Out = torch.empty_like(C)
        BLOCK_N = _next_pow2(N)
        norm_residual_kernel[(M,)](
            C, y, Out,
            N,
            BLOCK_N=BLOCK_N,
            EPS=self.eps,
        )
        return Out