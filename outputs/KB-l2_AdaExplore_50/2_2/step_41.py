import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SP': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 2048}, num_warps=8, num_stages=3),
    ],
    key=['SP'],
)
@triton.jit
def post_process_kernel(
    x_ptr, b_ptr, out_ptr,
    N, C, SP,
    SCALE: tl.constexpr,
    INV_SCALE: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_sp = tl.program_id(2)

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < SP

    base = pid_n * (C * SP) + pid_c * SP
    bias = tl.load(b_ptr + pid_c)

    vals = tl.load(x_ptr + base + sp_offs, mask=sp_mask, other=0.0)
    vals = vals + bias
    vals = tl.minimum(tl.maximum(vals, 0.0), 1.0)
    vals = vals * SCALE
    vals = tl.minimum(tl.maximum(vals, 0.0), 1.0)
    vals = vals * INV_SCALE
    tl.store(out_ptr + base + sp_offs, vals, mask=sp_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        # Use cuDNN for the heavy transposed convolution
        y = self.conv_transpose(x)
        N, C, H, W = y.shape
        SP = H * W

        # Fuse conv bias is already inside y. Add user bias and post-process in one kernel.
        fused_bias = self.bias.view(-1).contiguous()

        out = torch.empty_like(y)
        y = y.contiguous()

        grid = lambda meta: (C, N, triton.cdiv(SP, meta['BLOCK_SP']))
        post_process_kernel[grid](
            y, fused_bias, out,
            N, C, SP,
            float(self.scaling_factor),
            float(1.0 / self.scaling_factor),
        )
        return out