import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_post_kernel_cl(
    x_ptr,        # channels_last_3d: logical [N,C,D,H,W], memory [N,D,H,W,C]
    out_ptr,      # [total] contiguous output
    bias_ptr,     # scalar
    total, C,
    BLOCK_C: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS
    rows = row_start + tl.arange(0, ROWS)
    row_mask = rows < total

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    # ptrs: [ROWS, BLOCK_C]
    ptrs = x_ptr + rows[:, None] * C + c_offs[None, :]
    mask = row_mask[:, None] & c_mask[None, :]
    vals = tl.load(ptrs, mask=mask, other=-float('inf'))

    m = tl.max(vals, axis=1)
    e = tl.exp(vals - m[:, None])
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=1)
    lse = m + tl.log(s)

    sig = 1.0 / (1.0 + tl.exp(-(lse + 3.0)))
    hs = lse * sig / 6.0

    b = tl.load(bias_ptr)
    y = hs - b
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)

    tl.store(out_ptr + rows, y, mask=row_mask)


def fused_post(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # x is channels_last_3d, logical shape [N, C, D, H, W]
    N, C, D, H, W = x.shape
    total = N * D * H * W
    BLOCK_C = triton.next_power_of_2(C)
    if BLOCK_C < 16:
        BLOCK_C = 16

    # Output as contiguous [N, 1, D, H, W]
    out = torch.empty((N, 1, D, H, W), device=x.device, dtype=x.dtype)
    out_flat = out.view(-1)

    # Get a flat view of x in memory order [N, D, H, W, C]
    # x.permute(0,2,3,4,1).contiguous() would copy; instead we rely on the
    # channels_last_3d strides and just pass the raw pointer + total + C.
    ROWS = 8
    grid = (triton.cdiv(total, ROWS),)
    fused_post_kernel_cl[grid](
        x, out_flat, bias.reshape(-1),
        total, C,
        BLOCK_C=BLOCK_C,
        ROWS=ROWS,
        num_warps=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))

    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last_3d)
        x = self.conv_transpose(x)
        # Ensure channels_last_3d layout for fast C-contiguous reduction
        if not x.is_contiguous(memory_format=torch.channels_last_3d):
            x = x.contiguous(memory_format=torch.channels_last_3d)
        x = fused_post(x, self.bias)
        return x