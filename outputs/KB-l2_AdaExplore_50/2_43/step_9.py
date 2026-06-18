import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_pool_lse_relu_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    Dp, Hp, Wp,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    BLOCK_C: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # one program processes BLOCK_W consecutive wp positions for a given (n, dp, hp)
    pid_w = tl.program_id(0)
    pid_hd = tl.program_id(1)
    pid_n = tl.program_id(2)

    hp = pid_hd % Hp
    dp = pid_hd // Hp
    n = pid_n

    wp_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = wp_offs < Wp

    d0 = dp * 2
    h0 = hp * 2
    w0 = wp_offs * 2  # [BLOCK_W]

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    # 2D shape: [BLOCK_C, BLOCK_W]
    base = n * stride_n + offs_c[:, None] * stride_c + w0[None, :] * stride_w
    mask2d = mask_c[:, None] & mask_w[None, :]

    off_d0 = (d0 + 0) * stride_d
    off_d1 = (d0 + 1) * stride_d
    off_h0 = (h0 + 0) * stride_h
    off_h1 = (h0 + 1) * stride_h
    sw = stride_w

    p000 = tl.load(x_ptr + base + off_d0 + off_h0,      mask=mask2d, other=-float('inf'))
    p001 = tl.load(x_ptr + base + off_d0 + off_h0 + sw, mask=mask2d, other=-float('inf'))
    p010 = tl.load(x_ptr + base + off_d0 + off_h1,      mask=mask2d, other=-float('inf'))
    p011 = tl.load(x_ptr + base + off_d0 + off_h1 + sw, mask=mask2d, other=-float('inf'))
    p100 = tl.load(x_ptr + base + off_d1 + off_h0,      mask=mask2d, other=-float('inf'))
    p101 = tl.load(x_ptr + base + off_d1 + off_h0 + sw, mask=mask2d, other=-float('inf'))
    p110 = tl.load(x_ptr + base + off_d1 + off_h1,      mask=mask2d, other=-float('inf'))
    p111 = tl.load(x_ptr + base + off_d1 + off_h1 + sw, mask=mask2d, other=-float('inf'))

    m1 = tl.maximum(p000, p001)
    m2 = tl.maximum(p010, p011)
    m3 = tl.maximum(p100, p101)
    m4 = tl.maximum(p110, p111)
    m5 = tl.maximum(m1, m2)
    m6 = tl.maximum(m3, m4)
    pooled = tl.maximum(m5, m6)  # [BLOCK_C, BLOCK_W]

    pooled_masked = tl.where(mask_c[:, None], pooled, -float('inf'))
    row_max = tl.max(pooled_masked, axis=0)  # [BLOCK_W]
    exps = tl.exp(pooled_masked - row_max[None, :])
    exps = tl.where(mask_c[:, None], exps, 0.0)
    sum_exp = tl.sum(exps, axis=0)  # [BLOCK_W]
    lse = tl.log(sum_exp) + row_max

    out_val = tl.maximum(lse, 0.0)

    out_offset = n * (Dp * Hp * Wp) + dp * (Hp * Wp) + hp * Wp + wp_offs
    tl.store(out_ptr + out_offset, out_val, mask=mask_w)


def fused_pool_lse_relu(x):
    N, C, D, H, W = x.shape
    Dp = D // 2
    Hp = H // 2
    Wp = W // 2
    x = x.contiguous()
    out = torch.empty((N, 1, Dp, Hp, Wp), device=x.device, dtype=x.dtype)

    # next power of 2 for C
    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    BLOCK_W = 64
    grid = (triton.cdiv(Wp, BLOCK_W), Dp * Hp, N)
    fused_pool_lse_relu_kernel[grid](
        x, out,
        N, C, D, H, W,
        Dp, Hp, Wp,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        BLOCK_C=BLOCK_C,
        BLOCK_W=BLOCK_W,
        num_warps=4,
        num_stages=3,
    )
    return out


# Enable cuDNN benchmark and TF32 for faster conv3d
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Convert to channels_last_3d for faster conv3d on Ampere
        self.conv = self.conv.to(memory_format=torch.channels_last_3d)

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last_3d)
        x = self.conv(x)
        x = x.contiguous()
        x = fused_pool_lse_relu(x)
        return x