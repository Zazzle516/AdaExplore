import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_act_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Mish: x * tanh(softplus(x))
    sp = tl.log(1.0 + tl.exp(x))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    mish = x * th
    e2m = tl.exp(2.0 * mish)
    out = (e2m - 1.0) / (e2m + 1.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_mish_tanh(x: torch.Tensor) -> torch.Tensor:
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    n = x_c.numel()
    BLOCK_SIZE = 8192
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_act_kernel[grid](x_c, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=2)
    return out


torch.backends.cudnn.benchmark = True


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Convert weight to channels_last_3d explicitly
        self.conv.weight.data = self.conv.weight.data.to(memory_format=torch.channels_last_3d)

    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last_3d)
        x = self.conv(x)
        x = fused_mish_tanh(x)
        return x