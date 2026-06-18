import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _bn_mean_sub_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,  # [C]
    shift_ptr,  # [C]
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)  # over N*C
    n = pid // C
    c = pid % C
    row_offset = pid * S

    scale = tl.load(scale_ptr + c)
    shift = tl.load(shift_ptr + c)

    # Pass 1: compute sum of (scale*x + shift)
    acc = 0.0
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        vals = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0)
        y = vals * scale + shift
        acc += tl.sum(tl.where(mask, y, 0.0), axis=0)
    mean = acc / S

    # Pass 2: write (y - mean)
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        vals = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0)
        y = vals * scale + shift - mean
        tl.store(out_ptr + row_offset + offs, y, mask=mask)


def bn_mean_sub(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.is_contiguous()
    N, C, D, H, W = x.shape
    S = D * H * W
    out = torch.empty_like(x)
    grid = (N * C,)
    if S <= 2048:
        BLOCK_S = 1024
        num_warps = 4
    elif S <= 8192:
        BLOCK_S = 2048
        num_warps = 8
    else:
        BLOCK_S = 4096
        num_warps = 8
    _bn_mean_sub_kernel[grid](
        x, out, scale, shift, N, C, S, BLOCK_S=BLOCK_S, num_warps=num_warps
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fuse BN (eval mode uses running stats; train mode uses batch stats).
        # For simplicity and correctness across modes, just call batch_norm normally.
        x = self.batch_norm(x)
        x = x.contiguous()
        # mean-subtract via fused kernel (scale=1, shift=0 here since BN already applied)
        N, C, D, H, W = x.shape
        scale = torch.ones(C, device=x.device, dtype=x.dtype)
        shift = torch.zeros(C, device=x.device, dtype=x.dtype)
        return bn_mean_sub(x, scale, shift)