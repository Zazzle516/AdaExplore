import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_epilogue_kernel(
    x_ptr,      # conv output, shape [N, C, D, H, W] contiguous
    bias_ptr,   # bias, shape [C]
    out_ptr,    # output
    DHW,
    NC,
    BLOCK_SIZE: tl.constexpr,
):
    # program over (nc, spatial_tile)
    nc = tl.program_id(0)
    sp = tl.program_id(1)

    # channel index for this program
    # nc = n*C + c, channel = nc % C. But we only need bias[c], which equals bias[nc % C].
    # Simpler: bias is broadcast per channel; we pass bias indexed by nc % C via a small mod.
    # Since C is constexpr-like (passed as arg), do mod here once per program.
    # Actually we need C; load bias by computing channel via nc and C passed in.
    # We'll load a single scalar.
    base = nc.to(tl.int64) * DHW + sp * BLOCK_SIZE
    offs = base + tl.arange(0, BLOCK_SIZE)
    mask = (sp * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) < DHW

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # bias index = nc % C, but we passed bias already gathered? No - pass channel via separate ptr indexing
    # We need C. Use nc and compute mod with C passed via separate kernel using NC unused.
    # Pass channel idx by using nc directly indexed into a bias array of length NC where bias is tiled.
    # Simpler: pass already-broadcast bias of length NC (one scalar per (n,c)).
    b = tl.load(bias_ptr + nc)
    out = (2.0 * x + b) * x + x
    tl.store(out_ptr + offs, out, mask=mask)


def fused_epilogue(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    N, C, D, H, W = x.shape
    DHW = D * H * W
    NC = N * C
    # build per-(n,c) bias by broadcasting bias[C] -> [N,C] flat
    bias_flat = bias.contiguous().view(C)
    bias_nc = bias_flat.unsqueeze(0).expand(N, C).contiguous().view(-1)
    out = torch.empty_like(x)
    BLOCK_SIZE = 2048
    grid = (NC, (DHW + BLOCK_SIZE - 1) // BLOCK_SIZE)
    fused_epilogue_kernel[grid](
        x, bias_nc, out, DHW, NC, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8, num_stages=3,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        return fused_epilogue(x, self.bias)