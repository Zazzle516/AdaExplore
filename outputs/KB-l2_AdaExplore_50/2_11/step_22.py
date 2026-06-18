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

    scale = tl.load(scale_ptr + c_offs)  # [C_PER_G]
    shift = tl.load(shift_ptr + c_offs)
    gw = tl.load(gn_weight_ptr + c_offs)
    gb = tl.load(gn_bias_ptr + c_offs)

    # 2D indices: [C_PER_G, SPATIAL]
    base_n = pid_n * C * H * W
    c_base = c_offs[:, None] * (H * W) + base_n
    p00_off = c_base + (ih[None, :] * W + iw[None, :])
    p01_off = c_base + (ih[None, :] * W + (iw[None, :] + 1))
    p10_off = c_base + ((ih[None, :] + 1) * W + iw[None, :])
    p11_off = c_base + ((ih[None, :] + 1) * W + (iw[None, :] + 1))

    p00 = tl.load(x_ptr + p00_off)
    p01 = tl.load(x_ptr + p01_off)
    p10 = tl.load(x_ptr + p10_off)
    p11 = tl.load(x_ptr + p11_off)

    s2 = scale[:, None]
    sh2 = shift[:, None]

    t00 = tl.extra.cuda.libdevice.tanh(p00 * s2 + sh2)
    t01 = tl.extra.cuda.libdevice.tanh(p01 * s2 + sh2)
    t10 = tl.extra.cuda.libdevice.tanh(p10 * s2 + sh2)
    t11 = tl.extra.cuda.libdevice.tanh(p11 * s2 + sh2)

    m1 = tl.maximum(t00, t01)
    m2 = tl.maximum(t10, t11)
    pooled = tl.maximum(m1, m2)  # [C_PER_G, SPATIAL]

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
            num_warps=8,
            num_stages=2,
        )
        return out