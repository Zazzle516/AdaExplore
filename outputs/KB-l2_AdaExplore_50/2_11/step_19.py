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
    scale_ptr, shift_ptr,
    gn_weight_ptr, gn_bias_ptr,
    N, C, H, W,
    H_out, W_out,
    G,
    eps,
    BLOCK_SPATIAL: tl.constexpr,
    C_PER_G_CONST: tl.constexpr,
):
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
    c_abs = pid_g * C_PER_G_CONST + c_offs

    scale = tl.load(scale_ptr + c_abs)
    shift = tl.load(shift_ptr + c_abs)

    n_offset = pid_n * C * H * W
    c_base = c_abs[:, None] * (H * W) + n_offset
    s00 = (ih * W + iw)[None, :]

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
    rstd = 1.0 / tl.sqrt(var + eps)

    gw = tl.load(gn_weight_ptr + c_abs)[:, None]
    gb = tl.load(gn_bias_ptr + c_abs)[:, None]

    normed = (pooled - mean) * rstd * gw + gb

    out_n = pid_n * C * H_out * W_out
    out_offs = c_abs[:, None] * (H_out * W_out) + offs[None, :] + out_n
    tl.store(out_ptr + out_offs, normed, mask=full_mask)


# ConvTranspose2d as a GEMM with implicit im2col.
# Weight layout: original is [IC, OC, KH, KW]. We reshape it (offline at first call) to
# [OC, IC*KH*KW] for a clean GEMM K-dim.
#
# For each output position (oh, ow):
#   y[n, oc, oh, ow] = sum_{ic, kh, kw} x[n, ic, oh+pad-kh, ow+pad-kw] * w[ic, oc, kh, kw]
#
# So K-dim is IC*KH*KW. We tile (N, OC_block, SP_block) and accumulate over K.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['IC', 'OC', 'H_OUT', 'W_OUT', 'KH', 'KW'],
)
@triton.jit
def conv_transpose2d_gemm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, IC, H, W,
    OC, H_OUT, W_OUT,
    KH: tl.constexpr, KW: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]
    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (H_OUT * W_OUT)

    oh = sp_offs // W_OUT  # [BLOCK_SP]
    ow = sp_offs % W_OUT

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    K_TOTAL = IC * KH * KW
    KHKW = KH * KW
    HW = H * W

    x_base_n = pid_n * IC * HW

    # Loop over K dimension in blocks
    for k_start in range(0, K_TOTAL, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_offs < K_TOTAL

        # Decompose k = ic * KH*KW + kh * KW + kw
        ic = k_offs // KHKW
        rem = k_offs % KHKW
        kh = rem // KW
        kw = rem % KW

        # x indices: ih = oh + PAD - kh, iw = ow + PAD - kw
        ih = oh[:, None] + PAD - kh[None, :]  # [BLOCK_SP, BLOCK_K]
        iw = ow[:, None] + PAD - kw[None, :]
        ih_valid = (ih >= 0) & (ih < H)
        iw_valid = (iw >= 0) & (iw < W)
        valid = ih_valid & iw_valid

        ih_safe = tl.where(ih_valid, ih, 0)
        iw_safe = tl.where(iw_valid, iw, 0)

        x_off = x_base_n + ic[None, :] * HW + ih_safe * W + iw_safe  # [BLOCK_SP, BLOCK_K]
        x_mask = valid & sp_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [BLOCK_SP, BLOCK_K]

        # w_ptr layout: [OC, IC*KH*KW]  (transposed/repacked)
        w_off = oc_offs[:, None] * K_TOTAL + k_offs[None, :]  # [BLOCK_OC, BLOCK_K]
        w_mask = oc_mask[:, None] & k_mask[None, :]
        w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # [BLOCK_OC, BLOCK_K]

        # acc[oc, sp] += sum_k w[oc, k] * x[sp, k]
        acc += tl.dot(w_tile, tl.trans(x_tile))

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += bias[:, None]

    y_off = pid_n * OC * H_OUT * W_OUT + oc_offs[:, None] * (H_OUT * W_OUT) + sp_offs[None, :]
    mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(y_ptr + y_off, acc, mask=mask)


def triton_conv_transpose2d_gemm(x, w_packed, bias, padding, kernel_size):
    N, IC, H, W = x.shape
    OC = w_packed.shape[0]
    KH = KW = kernel_size
    H_OUT = H - 1 + KH - 2 * padding
    W_OUT = W - 1 + KW - 2 * padding

    y = torch.empty((N, OC, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        N,
        triton.cdiv(OC, meta['BLOCK_OC']),
        triton.cdiv(H_OUT * W_OUT, meta['BLOCK_SP']),
    )
    conv_transpose2d_gemm_kernel[grid](
        x, w_packed, bias, y,
        N, IC, H, W,
        OC, H_OUT, W_OUT,
        KH=KH, KW=KW,
        PAD=padding,
    )
    return y


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
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self._packed_weight = None

    def _get_packed_weight(self):
        # original [IC, OC, KH, KW] -> [OC, IC*KH*KW]
        w = self.conv_transpose.weight  # [IC, OC, KH, KW]
        IC, OC, KH, KW = w.shape
        # permute to [OC, IC, KH, KW] then reshape
        w_packed = w.permute(1, 0, 2, 3).contiguous().view(OC, IC * KH * KW)
        return w_packed

    def forward(self, x):
        if self.training or self.batch_norm.running_mean is None:
            x = self.conv_transpose(x)
            x = self.batch_norm(x)
            x = self.tanh(x)
            x = self.max_pool(x)
            x = self.group_norm(x)
            return x

        x = x.contiguous()

        if self._packed_weight is None or self._packed_weight.device != x.device:
            self._packed_weight = self._get_packed_weight().to(x.device).contiguous()
        w_packed = self._packed_weight

        b = self.conv_transpose.bias
        if b is None:
            b = torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)
        else:
            b = b.contiguous()

        if self.stride == 1:
            x = triton_conv_transpose2d_gemm(x, w_packed, b, self.padding, self.kernel_size)
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
            G,
            float(self.group_norm.eps),
            BLOCK_SPATIAL=BLOCK_SPATIAL,
            C_PER_G_CONST=C_per_G,
        )
        return out