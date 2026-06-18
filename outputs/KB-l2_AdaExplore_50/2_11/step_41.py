import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=['C_PER_G_CONST', 'BLOCK_SPATIAL'],
)
@triton.jit
def fused_bn_tanh_maxpool_gn_kernel(
    x_ptr, out_ptr,
    scale_ptr, shift_ptr,  # BN folded: scale[C], shift[C]
    gn_weight_ptr, gn_bias_ptr,  # GN affine: [C]
    N, C, H, W,  # input dims (before pooling)
    H_out, W_out,  # output dims (after pooling)
    G, C_PER_G,  # groups, channels per group
    eps,
    BLOCK_SPATIAL: tl.constexpr,
    C_PER_G_CONST: tl.constexpr,
):
    # one program per (n, g)
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    spatial_size = H_out * W_out
    group_size = C_PER_G_CONST * spatial_size

    offs = tl.arange(0, BLOCK_SPATIAL)
    mask = offs < spatial_size
    oh = offs // W_out
    ow = offs % W_out
    ih = oh * 2
    iw = ow * 2

    sum_val = tl.zeros([], dtype=tl.float32)
    sum_sq = tl.zeros([], dtype=tl.float32)

    # Cache pooled values across channels in the group
    # shape: [C_PER_G_CONST, BLOCK_SPATIAL]
    c_range = tl.arange(0, C_PER_G_CONST)
    c_global = pid_g * C_PER_G_CONST + c_range  # [C_PER_G_CONST]

    scale_v = tl.load(scale_ptr + c_global)  # [C_PER_G_CONST]
    shift_v = tl.load(shift_ptr + c_global)  # [C_PER_G_CONST]

    # base offsets per channel
    base_n = pid_n * C * H * W
    base_c = c_global * (H * W) + base_n  # [C_PER_G_CONST]

    # 2D offsets: [C_PER_G_CONST, BLOCK_SPATIAL]
    base_2d = base_c[:, None]
    ih_2d = ih[None, :]
    iw_2d = iw[None, :]
    mask_2d = mask[None, :]

    addr00 = base_2d + ih_2d * W + iw_2d
    addr01 = addr00 + 1
    addr10 = addr00 + W
    addr11 = addr10 + 1

    p00 = tl.load(x_ptr + addr00, mask=mask_2d, other=-float('inf'))
    p01 = tl.load(x_ptr + addr01, mask=mask_2d, other=-float('inf'))
    p10 = tl.load(x_ptr + addr10, mask=mask_2d, other=-float('inf'))
    p11 = tl.load(x_ptr + addr11, mask=mask_2d, other=-float('inf'))

    sc = scale_v[:, None]
    sh = shift_v[:, None]

    t00 = tl.extra.cuda.libdevice.tanh(p00 * sc + sh)
    t01 = tl.extra.cuda.libdevice.tanh(p01 * sc + sh)
    t10 = tl.extra.cuda.libdevice.tanh(p10 * sc + sh)
    t11 = tl.extra.cuda.libdevice.tanh(p11 * sc + sh)

    m1 = tl.maximum(t00, t01)
    m2 = tl.maximum(t10, t11)
    pooled = tl.maximum(m1, m2)  # [C_PER_G_CONST, BLOCK_SPATIAL]

    pooled_masked = tl.where(mask_2d, pooled, 0.0)
    sum_val = tl.sum(pooled_masked)
    sum_sq = tl.sum(pooled_masked * pooled_masked)

    mean = sum_val / group_size
    var = sum_sq / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    gw = tl.load(gn_weight_ptr + c_global)[:, None]
    gb = tl.load(gn_bias_ptr + c_global)[:, None]

    normed = (pooled - mean) * rstd * gw + gb

    out_base_n = pid_n * C * H_out * W_out
    out_base_c = c_global * (H_out * W_out) + out_base_n  # [C_PER_G_CONST]
    out_addr = out_base_c[:, None] + offs[None, :]

    tl.store(out_ptr + out_addr, normed, mask=mask_2d)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)

        # Compute BN folded scale/shift
        bn = self.batch_norm
        if self.training or bn.running_mean is None:
            # fall back to torch for training
            x = bn(x)
            x = torch.tanh(x)
            x = self.max_pool(x)
            x = self.group_norm(x)
            return x

        running_mean = bn.running_mean
        running_var = bn.running_var
        bn_w = bn.weight
        bn_b = bn.bias
        bn_eps = bn.eps

        inv_std = torch.rsqrt(running_var + bn_eps)
        scale = bn_w * inv_std
        shift = bn_b - running_mean * scale

        N, C, H, W = x.shape
        H_out = H // 2
        W_out = W // 2
        G = self.num_groups
        C_per_G = C // G

        x = x.contiguous()
        out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

        spatial = H_out * W_out
        BLOCK_SPATIAL = triton.next_power_of_2(spatial)
        if BLOCK_SPATIAL < 16:
            BLOCK_SPATIAL = 16

        grid = (N, G)
        fused_bn_tanh_maxpool_gn_kernel[grid](
            x, out,
            scale.contiguous(), shift.contiguous(),
            self.group_norm.weight.contiguous(), self.group_norm.bias.contiguous(),
            N, C, H, W,
            H_out, W_out,
            G, C_per_G,
            float(self.group_norm.eps),
            BLOCK_SPATIAL=BLOCK_SPATIAL,
            C_PER_G_CONST=C_per_G,
        )
        return out