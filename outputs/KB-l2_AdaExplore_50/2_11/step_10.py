import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


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

    # First pass: compute pooled+tanh activations, accumulate sum and sum_sq
    sum_val = tl.zeros([], dtype=tl.float32)
    sum_sq = tl.zeros([], dtype=tl.float32)

    # iterate over channels in group and spatial blocks
    for c_local in tl.static_range(0, C_PER_G_CONST):
        c = pid_g * C_PER_G_CONST + c_local
        scale = tl.load(scale_ptr + c)
        shift = tl.load(shift_ptr + c)

        for s_start in range(0, spatial_size, BLOCK_SPATIAL):
            offs = s_start + tl.arange(0, BLOCK_SPATIAL)
            mask = offs < spatial_size
            oh = offs // W_out
            ow = offs % W_out
            ih = oh * 2
            iw = ow * 2

            base = pid_n * C * H * W + c * H * W
            # load 2x2 window
            p00 = tl.load(x_ptr + base + ih * W + iw, mask=mask, other=-float('inf'))
            p01 = tl.load(x_ptr + base + ih * W + (iw + 1), mask=mask, other=-float('inf'))
            p10 = tl.load(x_ptr + base + (ih + 1) * W + iw, mask=mask, other=-float('inf'))
            p11 = tl.load(x_ptr + base + (ih + 1) * W + (iw + 1), mask=mask, other=-float('inf'))

            # apply BN + tanh
            t00 = tl.extra.cuda.libdevice.tanh(p00 * scale + shift)
            t01 = tl.extra.cuda.libdevice.tanh(p01 * scale + shift)
            t10 = tl.extra.cuda.libdevice.tanh(p10 * scale + shift)
            t11 = tl.extra.cuda.libdevice.tanh(p11 * scale + shift)

            m1 = tl.maximum(t00, t01)
            m2 = tl.maximum(t10, t11)
            pooled = tl.maximum(m1, m2)

            pooled_masked = tl.where(mask, pooled, 0.0)
            sum_val += tl.sum(pooled_masked)
            sum_sq += tl.sum(pooled_masked * pooled_masked)

    mean = sum_val / group_size
    var = sum_sq / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: recompute and write output normalized
    for c_local in tl.static_range(0, C_PER_G_CONST):
        c = pid_g * C_PER_G_CONST + c_local
        scale = tl.load(scale_ptr + c)
        shift = tl.load(shift_ptr + c)
        gw = tl.load(gn_weight_ptr + c)
        gb = tl.load(gn_bias_ptr + c)

        for s_start in range(0, spatial_size, BLOCK_SPATIAL):
            offs = s_start + tl.arange(0, BLOCK_SPATIAL)
            mask = offs < spatial_size
            oh = offs // W_out
            ow = offs % W_out
            ih = oh * 2
            iw = ow * 2

            base = pid_n * C * H * W + c * H * W
            p00 = tl.load(x_ptr + base + ih * W + iw, mask=mask, other=-float('inf'))
            p01 = tl.load(x_ptr + base + ih * W + (iw + 1), mask=mask, other=-float('inf'))
            p10 = tl.load(x_ptr + base + (ih + 1) * W + iw, mask=mask, other=-float('inf'))
            p11 = tl.load(x_ptr + base + (ih + 1) * W + (iw + 1), mask=mask, other=-float('inf'))

            t00 = tl.extra.cuda.libdevice.tanh(p00 * scale + shift)
            t01 = tl.extra.cuda.libdevice.tanh(p01 * scale + shift)
            t10 = tl.extra.cuda.libdevice.tanh(p10 * scale + shift)
            t11 = tl.extra.cuda.libdevice.tanh(p11 * scale + shift)

            m1 = tl.maximum(t00, t01)
            m2 = tl.maximum(t10, t11)
            pooled = tl.maximum(m1, m2)

            normed = (pooled - mean) * rstd * gw + gb

            out_base = pid_n * C * H_out * W_out + c * H_out * W_out
            tl.store(out_ptr + out_base + offs, normed, mask=mask)


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
        BLOCK_SPATIAL = min(triton.next_power_of_2(spatial), 256)
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
            num_warps=4,
        )
        return out