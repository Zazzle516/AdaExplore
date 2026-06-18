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
        triton.Config({}, num_warps=4, num_stages=3),
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
    G,
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
    smask = offs < spatial_size
    oh = offs // W_out
    ow = offs % W_out
    ih = oh * 2
    iw = ow * 2

    c_offs = tl.arange(0, C_PER_G_CONST)
    c_abs = pid_g * C_PER_G_CONST + c_offs  # [C_PER_G]

    scale = tl.load(scale_ptr + c_abs)  # [C_PER_G]
    shift = tl.load(shift_ptr + c_abs)

    n_offset = pid_n * C * H * W
    c_base = c_abs[:, None] * (H * W) + n_offset  # [C_PER_G, 1]
    s00 = (ih * W + iw)[None, :]  # [1, BLOCK_SPATIAL]

    full_mask = smask[None, :]
    neg_inf = float('-inf')

    p00 = tl.load(x_ptr + c_base + s00, mask=full_mask, other=neg_inf)
    p01 = tl.load(x_ptr + c_base + s00 + 1, mask=full_mask, other=neg_inf)
    p10 = tl.load(x_ptr + c_base + s00 + W, mask=full_mask, other=neg_inf)
    p11 = tl.load(x_ptr + c_base + s00 + W + 1, mask=full_mask, other=neg_inf)

    scale2d = scale[:, None]
    shift2d = shift[:, None]

    t00 = tl.extra.cuda.libdevice.tanh(p00 * scale2d + shift2d)
    t01 = tl.extra.cuda.libdevice.tanh(p01 * scale2d + shift2d)
    t10 = tl.extra.cuda.libdevice.tanh(p10 * scale2d + shift2d)
    t11 = tl.extra.cuda.libdevice.tanh(p11 * scale2d + shift2d)

    pooled = tl.maximum(tl.maximum(t00, t01), tl.maximum(t10, t11))
    pooled = tl.where(full_mask, pooled, 0.0)

    sum_val = tl.sum(pooled)
    sum_sq = tl.sum(pooled * pooled)

    mean = sum_val / group_size
    var = sum_sq / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + EPS)

    gw = tl.load(gn_weight_ptr + c_abs)[:, None]
    gb = tl.load(gn_bias_ptr + c_abs)[:, None]

    normed = (pooled - mean) * rstd * gw + gb

    out_n = pid_n * C * H_out * W_out
    out_offs = c_abs[:, None] * (H_out * W_out) + offs[None, :] + out_n
    tl.store(out_ptr + out_offs, normed, mask=full_mask)


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
        self._cached_scale = None
        self._cached_shift = None
        self._cached_gw = None
        self._cached_gb = None

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

        if self._cached_scale is None or self._cached_scale.device != x.device:
            running_mean = bn.running_mean
            running_var = bn.running_var
            bn_w = bn.weight
            bn_b = bn.bias
            bn_eps = bn.eps

            inv_std = torch.rsqrt(running_var + bn_eps)
            scale = (bn_w * inv_std).contiguous()
            shift = (bn_b - running_mean * scale).contiguous()
            self._cached_scale = scale
            self._cached_shift = shift
            self._cached_gw = self.group_norm.weight.contiguous()
            self._cached_gb = self.group_norm.bias.contiguous()

        scale = self._cached_scale
        shift = self._cached_shift

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
            scale, shift,
            self._cached_gw, self._cached_gb,
            N, C, H, W,
            H_out, W_out,
            G,
            BLOCK_SPATIAL=BLOCK_SPATIAL,
            C_PER_G_CONST=C_per_G,
            EPS=float(self.group_norm.eps),
        )
        return out