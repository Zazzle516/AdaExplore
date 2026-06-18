import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_N': 8192}, num_warps=32, num_stages=2),
        triton.Config({'BLOCK_N': 8192}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_N': 8192}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def norm_residual_kernel(
    X_ptr, Y_ptr, Out_ptr,
    N,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    base = row * N
    x = tl.load(X_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    s = tl.sum(x, axis=0)
    mean = s / N
    xm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xm * xm, axis=0) / N
    inv = tl.rsqrt(var + EPS)
    xn = xm * inv

    out = (xn + y) * y
    tl.store(Out_ptr + base + offs, out, mask=mask)


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
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)

    def forward(self, x, y):
        x = x.contiguous()
        y = y.contiguous()
        if not x.is_cuda:
            x = x.cuda()
        if not y.is_cuda:
            y = y.cuda()
        W = self.bmm.weight
        b = self.bmm.bias
        if not W.is_cuda:
            self.bmm.weight.data = W.cuda()
            W = self.bmm.weight
        if not b.is_cuda:
            self.bmm.bias.data = b.cuda()
            b = self.bmm.bias

        M, K = x.shape
        N = W.shape[0]

        C = torch.addmm(b, x, W.t())

        Out = torch.empty_like(C)
        norm_residual_kernel[(M,)](
            C, y, Out,
            N,
            EPS=self.eps,
        )
        return Out