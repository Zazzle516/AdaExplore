import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=3),
    ],
    key=['S'],
)
@triton.jit
def _epilogue_kernel(
    x_ptr, bias_ptr, out_ptr,
    NC, S,
    constant_value, scaling_factor,
    BLOCK_S: tl.constexpr,
):
    # one program per (n*c, S-tile)
    pid_nc = tl.program_id(0)
    pid_s = tl.program_id(1)

    # channel index for bias is pid_nc % C; bias has shape (C,)
    # We pass bias indexed by channel; the caller flattens (N,C) so we need C via modulo.
    # Instead, pass C as a runtime arg.
    offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = offs < S

    x_base = pid_nc * S
    x = tl.load(x_ptr + x_base + offs, mask=mask, other=0.0)

    # bias is per-channel; channel = pid_nc % C, encoded via stride trick: caller passes bias_ptr already
    # offset by channel. Here we compute channel from pid_nc % C using a passed value.
    # To keep kernel simple we expect bias indexed by pid_nc already loaded outside; instead read scalar.
    # We use a separate small load:
    b = tl.load(bias_ptr + pid_nc)  # scalar broadcast

    y = tl.minimum(x, constant_value) + b
    y = y * scaling_factor

    tl.store(out_ptr + x_base + offs, y, mask=mask)


def fused_epilogue(conv_out, bias, constant_value, scaling_factor):
    # conv_out: [N, C, H, W], bias: [C, 1, 1]
    N, C, H, W = conv_out.shape
    S = H * W
    conv_out = conv_out.contiguous()
    out = torch.empty_like(conv_out)

    # Build per-(n,c) bias view: shape [N*C], each element repeated from bias[c]
    bias_flat = bias.contiguous().view(C)
    # Expand to [N, C] then flatten without materializing extra memory: use repeat
    # We pass bias as length N*C tensor (expand + contiguous costs little; C=128)
    bias_nc = bias_flat.unsqueeze(0).expand(N, C).contiguous().view(-1)

    NC = N * C
    grid = lambda meta: (NC, triton.cdiv(S, meta['BLOCK_S']))
    _epilogue_kernel[grid](
        conv_out, bias_nc, out,
        NC, S,
        float(constant_value), float(scaling_factor),
    )
    return out.view(N, C, H, W)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # heavy op: real cuDNN conv (full multiply-add count, full output shape)
        y = self.conv(x)
        # fused elementwise epilogue: min(c) + bias + scale
        return fused_epilogue(y, self.bias, self.constant_value, self.scaling_factor)