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
    ],
    key=['C_PER_G_CONST', 'BLOCK_SPATIAL'],
)
@triton.jit
def fused_bn_tanh_maxpool_gn_kernel(
    x_ptr, out_ptr,
    scale_ptr, shift_ptr,
    gn_weight_ptr, gn_bias_ptr,
    N, C, H, W,
    H_out, W_out,
    G, C_PER_G,
    eps,
    BLOCK_SPATIAL: tl.constexpr,
    C_PER_G_CONST: tl.constexpr,
):
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

    c_range = tl.arange(0, C_PER_G_CONST)
    c_global = pid_g * C_PER_G_CONST + c_range

    scale_v = tl.load(scale_ptr + c_global)
    shift_v = tl.load(shift_ptr + c_global)

    base_n = pid_n * C * H * W
    base_c = c_global * (H * W) + base_n

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
    pooled = tl.maximum(m1, m2)

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
    out_base_c = c_global * (H_out * W_out) + out_base_n
    out_addr = out_base_c[:, None] + offs[None, :]

    tl.store(out_ptr + out_addr, normed, mask=mask_2d)


# Conv transpose 2d kernel: stride=1, equivalent to conv with flipped weights
# and effective padding = kernel_size - 1 - padding.
# Output: (N, OC, H_out, W_out) where H_out = H_in + 2*(K-1) - 2*padding = H_in + K - 1 - 2*padding... 
# Actually for stride=1: H_out = H_in - 1 + K - 2*padding = H_in + K - 1 - 2*padding
# Standard: H_out = (H_in - 1)*stride - 2*padding + (K-1) + output_padding + 1
#         = H_in - 2*padding + K - 1
# For 32 + 5 - 1 - 2 = 34. Good.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 32}, num_warps=4, num_stages=3),
    ],
    key=['IC', 'K', 'PAD_EFF', 'H_in', 'W_in'],
)
@triton.jit
def conv_transpose_2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, OC,
    H_in, W_in,
    H_out, W_out,
    K: tl.constexpr,
    PAD_EFF: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # grid: (N, ceil(OC/BLOCK_OC), ceil(H_out*W_out/BLOCK_HW))
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    oc_mask = oc_offs < OC

    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)  # [BLOCK_HW]
    hw_mask = hw_offs < (H_out * W_out)
    oh = hw_offs // W_out
    ow = hw_offs % W_out

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    # Conv equivalent: out[oh,ow] = sum_{ic,kh,kw} W_flip[oc,ic,kh,kw] * x_pad[oh+kh, ow+kw]
    # where W_flip[oc,ic,kh,kw] = W_orig[ic,oc,K-1-kh,K-1-kw]
    # Input index: ih_in = oh + kh - PAD_EFF, iw_in = ow + kw - PAD_EFF
    # PAD_EFF = K - 1 - padding

    # weight layout: (IC, OC, K, K)
    # For each (kh, kw), load weight tile [BLOCK_OC, IC] and input tile [IC, BLOCK_HW]
    # then accumulate. But to simplify, loop over kh, kw and ic.

    for kh in tl.static_range(0, K):
        for kw in tl.static_range(0, K):
            ih_in = oh + kh - PAD_EFF  # [BLOCK_HW]
            iw_in = ow + kw - PAD_EFF  # [BLOCK_HW]
            in_valid = (ih_in >= 0) & (ih_in < H_in) & (iw_in >= 0) & (iw_in < W_in)  # [BLOCK_HW]

            # flipped indices into original weight
            kh_flip = K - 1 - kh
            kw_flip = K - 1 - kw

            for ic in range(0, IC):
                # weight: W_orig[ic, oc, kh_flip, kw_flip], shape stride (OC*K*K, K*K, K, 1)
                w_addr = ic * (OC * K * K) + oc_offs * (K * K) + kh_flip * K + kw_flip
                w_val = tl.load(w_ptr + w_addr, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                # input: x[n, ic, ih_in, iw_in]
                x_addr = pid_n * (IC * H_in * W_in) + ic * (H_in * W_in) + ih_in * W_in + iw_in
                x_val = tl.load(x_ptr + x_addr, mask=hw_mask & in_valid, other=0.0)  # [BLOCK_HW]

                acc += w_val[:, None] * x_val[None, :]

    # add bias
    bias_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += bias_val[:, None]

    # store
    out_addr = pid_n * (OC * H_out * W_out) + oc_offs[:, None] * (H_out * W_out) + hw_offs[None, :]
    out_mask = oc_mask[:, None] & hw_mask[None, :]
    tl.store(out_ptr + out_addr, acc, mask=out_mask)


def conv_transpose_2d_triton(x, weight, bias, padding, kernel_size):
    N, IC, H_in, W_in = x.shape
    _, OC, K, _ = weight.shape
    H_out = H_in + K - 1 - 2 * padding
    W_out = W_in + K - 1 - 2 * padding
    PAD_EFF = K - 1 - padding

    out = torch.empty((N, OC, H_out, W_out), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        N,
        triton.cdiv(OC, meta['BLOCK_OC']),
        triton.cdiv(H_out * W_out, meta['BLOCK_HW']),
    )

    conv_transpose_2d_kernel[grid](
        x, weight, bias, out,
        N, IC, OC,
        H_in, W_in,
        H_out, W_out,
        K=K, PAD_EFF=PAD_EFF,
    )
    return out


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
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

    def forward(self, x):
        if self.training or self.batch_norm.running_mean is None:
            x = self.conv_transpose(x)
            x = self.batch_norm(x)
            x = torch.tanh(x)
            x = self.max_pool(x)
            x = self.group_norm(x)
            return x

        # custom conv transpose (only valid for stride=1)
        if self.stride == 1:
            x = x.contiguous()
            x = conv_transpose_2d_triton(
                x,
                self.conv_transpose.weight.contiguous(),
                self.conv_transpose.bias.contiguous(),
                self.padding,
                self.kernel_size,
            )
        else:
            x = self.conv_transpose(x)

        bn = self.batch_norm
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