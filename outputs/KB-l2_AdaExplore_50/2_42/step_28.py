import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# We need to compute:
#   y = conv_transpose2d(x)           # [N, OC, H_out, W_out]
#   m = mean(y, dim=(2,3))            # [N, OC, 1, 1]
#   z = m + bias                      # [N, OC, 1, 1]
#   r = logsumexp(z, dim=1)           # [N, 1, 1, 1]
#   s = sum(r, dim=(2,3)) * 10        # [N, 1]
#
# Key insight: mean over the output spatial dims of a conv_transpose can be
# computed without forming the full output if we are careful, BUT the safety
# contract forbids skipping the heavy multiply-add. So we MUST run the full
# conv_transpose at runtime.
#
# Strategy:
# 1. Run conv_transpose2d via torch (cuDNN) — this is the heavy op.
# 2. Fuse mean + bias + logsumexp + sum + scale into a single Triton kernel
#    that reads y once.


# Kernel: reduce conv_transpose output to per-(N, OC) mean.
# One program per (n, oc), reduces over H*W.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=2),
    ],
    key=['HW'],
)
@triton.jit
def _mean_kernel(
    y_ptr,           # [N, OC, H, W]
    out_ptr,         # [N, OC]
    HW,
    stride_n, stride_c,
    OC,
    BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    base = pid_n * stride_n + pid_c * stride_c
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for off in range(0, HW, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < HW
        v = tl.load(y_ptr + base + idx, mask=mask, other=0.0)
        acc += v
    total = tl.sum(acc, axis=0)
    mean = total / HW
    tl.store(out_ptr + pid_n * OC + pid_c, mean)


# Kernel: logsumexp over OC for each N, then * 10.
# input m: [N, OC], bias: [OC]
# output: [N, 1]
@triton.jit
def _lse_kernel(
    m_ptr,           # [N, OC]
    bias_ptr,        # [OC]
    out_ptr,         # [N]
    OC,
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
        # Heavy op: real conv_transpose, fully materialized.
        y = self.conv_transpose(x)  # [N, OC, H_out, W_out]
        N, OC, H, W = y.shape
        HW = H * W

        y = y.contiguous()
        m = torch.empty((N, OC), device=y.device, dtype=torch.float32)

        BLOCK = 1024
        grid = (N, OC)
        _mean_kernel[grid](
            y, m, HW,
            y.stride(0), y.stride(1),
            BLOCK=BLOCK,
            num_warps=4,
        )

        out = torch.empty((N,), device=y.device, dtype=torch.float32)
        # Bias is [OC,1,1] — flatten
        bias_flat = self.bias.view(-1).contiguous()

        # pick BLOCK_OC as next pow2 >= OC
        BLOCK_OC = 1
        while BLOCK_OC < OC:
            BLOCK_OC *= 2

        _lse_kernel[(N,)](
            m, bias_flat, out, OC,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
        )

        return out.view(N, 1)