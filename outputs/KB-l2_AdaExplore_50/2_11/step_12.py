import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bn_tanh_maxpool_gn_kernel(
    x_ptr, out_ptr,
    scale_ptr, shift_ptr,
    gn_weight_ptr, gn_bias_ptr,
    N, C, H, W,
    H_out, W_out,
    eps,
    SPATIAL: tl.constexpr,
    C_PER_G_CONST: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_size = C_PER_G_CONST * SPATIAL

    offs = tl.arange(0, SPATIAL)
    oh = offs // W_out
    ow = offs % W_out
    ih = oh * 2
    iw = ow * 2

    c_offs = tl.arange(0, C_PER_G_CONST) + pid_g * C_PER_G_CONST

    scale = tl.load(scale_ptr + c_offs)
    shift = tl.load(shift_ptr + c_offs)
    gw = tl.load(gn_weight_ptr + c_offs)
    gb = tl.load(gn_bias_ptr + c_offs)

    base_n = pid_n * C * H * W
    c_base = c_offs[:, None] * (H * W) + base_n
    row0 = ih[None, :] * W + iw[None, :]
    row1 = row0 + W

    p00 = tl.load(x_ptr + c_base + row0)
    p01 = tl.load(x_ptr + c_base + row0 + 1)
    p10 = tl.load(x_ptr + c_base + row1)
    p11 = tl.load(x_ptr + c_base + row1 + 1)

    s2 = scale[:, None]
    sh2 = shift[:, None]

    a00 = p00 * s2 + sh2
    a01 = p01 * s2 + sh2
    a10 = p10 * s2 + sh2
    a11 = p11 * s2 + sh2

    # max over 2x2 BEFORE tanh (tanh is monotonic increasing)
    m1 = tl.maximum(a00, a01)
    m2 = tl.maximum(a10, a11)
    am = tl.maximum(m1, m2)
    pooled = tl.extra.cuda.libdevice.tanh(am)

    sum_val = tl.sum(pooled)
    sum_sq = tl.sum(pooled * pooled)

    mean = sum_val / group_size
    var = sum_sq / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    normed = (pooled - mean) * rstd * gw[:, None] + gb[:, None]

    out_base_n = pid_n * C * H_out * W_out
    out_off = out_base_n + c_offs[:, None] * (H_out * W_out) + offs[None, :]
    tl.store(out_ptr + out_off, normed)


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

        bn = self.batch_norm
        if self.training or bn.running_mean is None:
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
        scale = (bn_w * inv_std).contiguous()
        shift = (bn_b - running_mean * scale).contiguous()

        N, C, H, W = x.shape
        H_out = H // 2
        W_out = W // 2
        G = self.num_groups
        C_per_G = C // G

        x = x.contiguous()
        out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

        spatial = H_out * W_out
        SPATIAL = triton.next_power_of_2(spatial)

        grid = (N, G)
        fused_bn_tanh_maxpool_gn_kernel[grid](
            x, out,
            scale, shift,
            self.group_norm.weight.contiguous(), self.group_norm.bias.contiguous(),
            N, C, H, W,
            H_out, W_out,
            float(self.group_norm.eps),
            SPATIAL=SPATIAL,
            C_PER_G_CONST=C_per_G,
            num_warps=4,
        )
        return out