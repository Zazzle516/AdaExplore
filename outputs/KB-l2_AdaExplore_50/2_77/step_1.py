import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _global_avg_pool_kernel(
    x_ptr, out_ptr,
    N, C, S,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per (n, c)
    n = pid // C
    c = pid % C
    base = n * C * S + c * S
    acc = 0.0
    for s_start in range(0, S, BLOCK):
        offs = s_start + tl.arange(0, BLOCK)
        mask = offs < S
        vals = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
    tl.store(out_ptr + pid, acc / S)


def triton_global_avg_pool3d(x: torch.Tensor) -> torch.Tensor:
    # x: (N, C, D, H, W) -> (N, C, 1, 1, 1)
    N, C, D, H, W = x.shape
    S = D * H * W
    x_c = x.contiguous()
    out = torch.empty((N, C), device=x.device, dtype=x.dtype)
    BLOCK = 1024
    grid = (N * C,)
    _global_avg_pool_kernel[grid](x_c, out, N, C, S, BLOCK=BLOCK, num_warps=4)
    return out.view(N, C, 1, 1, 1)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x * self.scale_factor
        x = self.batch_norm(x)
        x = triton_global_avg_pool3d(x)
        return x