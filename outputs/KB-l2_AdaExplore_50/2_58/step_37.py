import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_post_kernel(
    x_ptr,        # [N, C, D, H, W] contiguous NCDHW
    out_ptr,      # [N, 1, D, H, W]
    bias_ptr,     # scalar
    SPATIAL,      # D*H*W
    NSPATIAL,     # N*D*H*W
    C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    s_offs = pid * BLOCK_S + tl.arange(0, BLOCK_S)  # [BLOCK_S]
    s_mask = s_offs < NSPATIAL

    # decode s_offs -> n, sp(=d*H*W + h*W + w)
    n = s_offs // SPATIAL
    sp = s_offs % SPATIAL

    base = n * (C * SPATIAL) + sp  # [BLOCK_S]

    c_offs = tl.arange(0, C)  # [C]
    # ptrs: [BLOCK_S, C]
    ptrs = x_ptr + base[:, None] + c_offs[None, :] * SPATIAL
    mask2d = s_mask[:, None]
    vals = tl.load(ptrs, mask=mask2d, other=-float('inf'))

    m = tl.max(vals, axis=1)              # [BLOCK_S]
    e = tl.exp(vals - m[:, None])
    s = tl.sum(e, axis=1)                 # [BLOCK_S]
    lse = m + tl.log(s)

    sig = 1.0 / (1.0 + tl.exp(-(lse + 3.0)))
    hs = lse * sig / 6.0

    b = tl.load(bias_ptr)
    y = hs - b
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)

    tl.store(out_ptr + s_offs, y, mask=s_mask)


def fused_post(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    N, C, D, H, W = x.shape
    x = x.contiguous()
    out = torch.empty((N, 1, D, H, W), device=x.device, dtype=x.dtype)
    SPATIAL = D * H * W
    NSPATIAL = N * SPATIAL
    BLOCK_S = 256
    grid = (triton.cdiv(NSPATIAL, BLOCK_S),)
    fused_post_kernel[grid](
        x, out, bias.reshape(-1),
        SPATIAL, NSPATIAL,
        C=C, BLOCK_S=BLOCK_S,
        num_warps=2,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_post(x, self.bias)
        return x