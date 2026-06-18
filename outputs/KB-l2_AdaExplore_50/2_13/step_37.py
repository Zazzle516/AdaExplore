import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_mean_post_kernel(
    x_ptr,         # (B, C, D, H, W) - conv output
    bias_ptr,      # (C,)
    out_ptr,       # (B, C, H, W)
    B, C, D, H, W,
    scaling_factor,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # one program per (b, h, w); fuse mean over D + bias + softmax(C) + tanh + scale
    pid = tl.program_id(0)
    HW = H * W
    b = pid // HW
    rem = pid % HW
    h = rem // W
    w = rem % W

    offs_c = tl.arange(0, BLOCK_C)
    offs_d = tl.arange(0, BLOCK_D)
    mask_c = offs_c < C
    mask_d = offs_d < D

    # base pointer at (b, 0, 0, h, w)
    base = b * C * D * HW + h * W + w
    # 2D pointers: [BLOCK_C, BLOCK_D]
    ptrs = x_ptr + base + offs_c[:, None] * (D * HW) + offs_d[None, :] * HW
    mask = mask_c[:, None] & mask_d[None, :]

    vals = tl.load(ptrs, mask=mask, other=0.0)
    s_d = tl.sum(vals, axis=1)  # [BLOCK_C]
    inv_D = 1.0 / D.to(tl.float32) if False else (1.0 / D)
    mean_c = s_d * inv_D

    bias = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
    x = mean_c + bias

    # softmax across channels
    x = tl.where(mask_c, x, -float('inf'))
    x_max = tl.max(x, axis=0)
    x_shift = x - x_max
    e = tl.exp(x_shift)
    e = tl.where(mask_c, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s

    # tanh via exp
    two_x = 2.0 * sm
    e2 = tl.exp(two_x)
    th = (e2 - 1.0) / (e2 + 1.0)
    out = th * scaling_factor

    out_base = b * C * HW + h * W + w
    out_ptrs = out_ptr + out_base + offs_c * HW
    tl.store(out_ptrs, out, mask=mask_c)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.out_channels = out_channels
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    def forward(self, x):
        x = self.conv_transpose(x)            # (B, C, D, H, W)
        x = x.contiguous()

        B, C, D, H, W = x.shape
        out = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype)

        # next power of 2 >= C
        BLOCK_C = 1
        while BLOCK_C < C:
            BLOCK_C *= 2
        BLOCK_D = 1
        while BLOCK_D < D:
            BLOCK_D *= 2

        bias_flat = self.bias.view(-1).contiguous()

        grid = (B * H * W,)
        fused_mean_post_kernel[grid](
            x, bias_flat, out,
            B, C, D, H, W,
            float(self.scaling_factor),
            BLOCK_C=BLOCK_C,
            BLOCK_D=BLOCK_D,
            num_warps=4,
        )
        # Match original: keepdim=True on dim=2
        return out.unsqueeze(2)