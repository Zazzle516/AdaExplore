import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=2, num_stages=3),
    ],
    key=['CPG', 'HW_OUT'],
)
@triton.jit
def fused_bn_tanh_pool_gn_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    shift_ptr,
    gn_weight_ptr,
    gn_bias_ptr,
    N, C, H, W,
    H_out, W_out,
    G,
    eps,
    CPG: tl.constexpr,
    HW_OUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    total = CPG * HW_OUT
    c_base = g * CPG

    offs_c = tl.arange(0, CPG)
    offs_s = tl.arange(0, BLOCK)
    mask_s = offs_s < HW_OUT

    scale = tl.load(scale_ptr + c_base + offs_c)
    shift = tl.load(shift_ptr + c_base + offs_c)
    gw = tl.load(gn_weight_ptr + c_base + offs_c)
    gb = tl.load(gn_bias_ptr + c_base + offs_c)

    oh = offs_s // W_out
    ow = offs_s % W_out
    ih = oh * 2
    iw = ow * 2

    n_off = n * C * H * W
    c_stride = H * W
    c_off = (c_base + offs_c)[:, None] * c_stride

    s00 = (ih * W + iw)[None, :]
    s01 = s00 + 1
    s10 = s00 + W
    s11 = s10 + 1

    mask2 = mask_s[None, :]

    p00 = tl.load(x_ptr + n_off + c_off + s00, mask=mask2, other=-1e30)
    p01 = tl.load(x_ptr + n_off + c_off + s01, mask=mask2, other=-1e30)
    p10 = tl.load(x_ptr + n_off + c_off + s10, mask=mask2, other=-1e30)
    p11 = tl.load(x_ptr + n_off + c_off + s11, mask=mask2, other=-1e30)

    sc = scale[:, None]
    sh = shift[:, None]

    # Take max in pre-bn space if scale > 0; but to be safe and correct, do bn first.
    v00 = p00 * sc + sh
    v01 = p01 * sc + sh
    v10 = p10 * sc + sh
    v11 = p11 * sc + sh

    # max before tanh (tanh is monotonic increasing) - saves 3 tanh computations per pooled pixel
    m0 = tl.maximum(v00, v01)
    m1 = tl.maximum(v10, v11)
    mx = tl.maximum(m0, m1)

    # tanh = 2*sigmoid(2x) - 1
    pooled = 2.0 * tl.sigmoid(2.0 * mx) - 1.0

    pooled_m = tl.where(mask2, pooled, 0.0)
    sum_val = tl.sum(pooled_m)
    sum_sq = tl.sum(pooled_m * pooled_m)

    inv_total = 1.0 / total
    mean = sum_val * inv_total
    var = sum_sq * inv_total - mean * mean
    rstd = tl.rsqrt(var + eps)

    normed = (pooled - mean) * rstd * gw[:, None] + gb[:, None]

    out_n_off = n * C * H_out * W_out
    out_c_stride = H_out * W_out
    out_c_off = (c_base + offs_c)[:, None] * out_c_stride
    out_s_off = offs_s[None, :]

    tl.store(out_ptr + out_n_off + out_c_off + out_s_off, normed, mask=mask2)


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
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def forward(self, x):
        x = self.conv_transpose(x)

        bn = self.batch_norm
        if bn.training:
            x = bn(x)
            x = torch.tanh(x)
            x = self.max_pool(x)
            x = self.group_norm(x)
            return x

        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        shift = bn.bias - bn.running_mean * scale

        # Note: the maximum-before-tanh trick requires scale >= 0 to be exactly equivalent.
        # With absolute(scale), maximum(a*scale, b*scale) = scale * maximum(a,b) only when scale>=0.
        # If any scale < 0, we need to handle it. Check and fall back if needed.
        if (scale < 0).any().item():
            # Fall back: do tanh then max (mathematically equivalent and tanh is monotonic so
            # actually max-then-tanh works regardless of scale sign because tanh is monotonic
            # and BN is affine - but max(p*s+sh) for s<0 is min(p)*s+sh, not max(p)*s+sh).
            # So we need actual computation. Use a separate path:
            pass

        N, C, H, W = x.shape
        H_out = H // 2
        W_out = W // 2
        G = self.num_groups
        CPG = C // G
        HW_OUT = H_out * W_out

        x = x.contiguous()
        out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

        BLOCK = 1
        while BLOCK < HW_OUT:
            BLOCK *= 2
        if BLOCK < 16:
            BLOCK = 16

        # Check if all scales are non-negative (common case for BN with positive gamma)
        all_nonneg = bool((scale >= 0).all().item())

        if all_nonneg:
            grid = (N * G,)
            fused_bn_tanh_pool_gn_kernel[grid](
                x, out,
                scale.contiguous(), shift.contiguous(),
                self.group_norm.weight.contiguous(), self.group_norm.bias.contiguous(),
                N, C, H, W,
                H_out, W_out,
                G,
                self.group_norm.eps,
                CPG=CPG,
                HW_OUT=HW_OUT,
                BLOCK=BLOCK,
            )
            return out
        else:
            # Fallback: standard order
            x_bn = x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
            x_bn = torch.tanh(x_bn)
            x_bn = F.max_pool2d(x_bn, 2, 2)
            return self.group_norm(x_bn)