import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=16, num_stages=2),
    ],
    key=['HW'],
)
@triton.jit
def _mean_kernel(
    y_ptr, out_ptr, HW,
    stride_n, stride_c, OC,
    BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    base = pid_n * stride_n + pid_c * stride_c
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    off = 0
    while off < HW:
        idx = off + tl.arange(0, BLOCK)
        mask = idx < HW
        v = tl.load(y_ptr + base + idx, mask=mask, other=0.0)
        acc += v
        off += BLOCK
    total = tl.sum(acc, axis=0)
    mean = total / HW.to(tl.float32)
    tl.store(out_ptr + pid_n * OC + pid_c, mean)


@triton.jit
def _lse_kernel(
    m_ptr, bias_ptr, out_ptr, OC,
    BLOCK_OC: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_OC)
    mask = offs < OC
    m = tl.load(m_ptr + pid * OC + offs, mask=mask, other=-float('inf'))
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
    z = m + b
    z_masked = tl.where(mask, z, -float('inf'))
    mx = tl.max(z_masked, axis=0)
    e = tl.exp(z_masked - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    lse = mx + tl.log(s)
    tl.store(out_ptr + pid, lse * 10.0)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.out_channels = out_channels

    def forward(self, x):
        y = self.conv_transpose(x)
        N, OC, H, W = y.shape
        HW = H * W
        y = y.contiguous()

        m = torch.empty((N, OC), device=y.device, dtype=torch.float32)
        _mean_kernel[(N, OC)](
            y, m, HW,
            y.stride(0), y.stride(1), OC,
        )

        out = torch.empty((N,), device=y.device, dtype=torch.float32)
        bias_flat = self.bias.view(-1).contiguous()

        BLOCK_OC = 1
        while BLOCK_OC < OC:
            BLOCK_OC *= 2

        _lse_kernel[(N,)](
            m, bias_flat, out, OC,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
        )

        return out.view(N, 1)