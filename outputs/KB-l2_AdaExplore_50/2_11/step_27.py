import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_bn_tanh_kernel(
    x_ptr,         # input [N, IC, IH, IW]
    w_ptr,         # weight (flipped, reshaped as conv2d) [OC, IC, K, K]
    scale_ptr,     # [OC]
    shift_ptr,     # [OC]
    out_ptr,       # [N, OC, OH, OW]
    N, IC, IH, IW,
    OC, OH, OW,
    K: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    IC_BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    offs_sp = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]

    mask_oc = offs_oc < OC
    mask_sp = offs_sp < (OH * OW)

    oh = offs_sp // OW
    ow = offs_sp % OW

    # Accumulator
    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # For each kernel position, for each input channel
    for kh in tl.static_range(0, K):
        ih = oh + kh - PAD  # [BLOCK_SP]
        ih_valid = (ih >= 0) & (ih < IH)
        for kw in tl.static_range(0, K):
            iw = ow + kw - PAD  # [BLOCK_SP]
            iw_valid = (iw >= 0) & (iw < IW)
            sp_valid = ih_valid & iw_valid  # [BLOCK_SP]

            # safe indices
            ih_s = tl.where(ih_valid, ih, 0)
            iw_s = tl.where(iw_valid, iw, 0)
            spatial_off = ih_s * IW + iw_s  # [BLOCK_SP]

            for ic_start in range(0, IC, IC_BLOCK):
                offs_ic = ic_start + tl.arange(0, IC_BLOCK)  # [IC_BLOCK]
                mask_ic = offs_ic < IC

                # Load input [IC_BLOCK, BLOCK_SP]
                x_off = (pid_n * IC * IH * IW
                         + offs_ic[:, None] * (IH * IW)
                         + spatial_off[None, :])
                x_mask = (mask_ic[:, None] & sp_valid[None, :] & mask_sp[None, :])
                x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [IC_BLOCK, BLOCK_SP]

                # Load weight [BLOCK_OC, IC_BLOCK]
                # weight layout: [OC, IC, K, K]
                w_off = (offs_oc[:, None] * (IC * K * K)
                         + offs_ic[None, :] * (K * K)
                         + kh * K + kw)
                w_mask = mask_oc[:, None] & mask_ic[None, :]
                w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # [BLOCK_OC, IC_BLOCK]

                acc += tl.dot(w_vals, x_vals)

    # Apply BN scale/shift
    scale = tl.load(scale_ptr + offs_oc, mask=mask_oc, other=0.0)
    shift = tl.load(shift_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc * scale[:, None] + shift[:, None]

    # Tanh
    acc = 2.0 * tl.sigmoid(2.0 * acc) - 1.0

    # Store [N, OC, OH, OW]
    out_off = (pid_n * OC * OH * OW
               + offs_oc[:, None] * (OH * OW)
               + offs_sp[None, :])
    out_mask = mask_oc[:, None] & mask_sp[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


@triton.jit
def pool_gn_kernel(
    x_ptr,           # input after conv-bn-tanh: [N, C, H, W]
    out_ptr,         # output: [N, C, H/2, W/2]
    gn_weight_ptr,   # [C]
    gn_bias_ptr,     # [C]
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
    s01 = (ih * W + (iw + 1))[None, :]
    s10 = ((ih + 1) * W + iw)[None, :]
    s11 = ((ih + 1) * W + (iw + 1))[None, :]

    mask2 = mask_s[None, :]

    p00 = tl.load(x_ptr + n_off + c_off + s00, mask=mask2, other=-1e30)
    p01 = tl.load(x_ptr + n_off + c_off + s01, mask=mask2, other=-1e30)
    p10 = tl.load(x_ptr + n_off + c_off + s10, mask=mask2, other=-1e30)
    p11 = tl.load(x_ptr + n_off + c_off + s11, mask=mask2, other=-1e30)

    m0 = tl.maximum(p00, p01)
    m1 = tl.maximum(p10, p11)
    pooled = tl.maximum(m0, m1)

    pooled_m = tl.where(mask2, pooled, 0.0)
    sum_val = tl.sum(pooled_m)
    sum_sq = tl.sum(pooled_m * pooled_m)

    mean = sum_val / total
    var = sum_sq / total - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

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
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        # Precompute the equivalent conv weight for stride=1 case
        # ConvTranspose2d weight shape: [IC, OC, K, K]
        # Equivalent Conv2d weight: spatially-flipped and transposed: [OC, IC, K, K]
        self._cached_weight = None

    def _get_conv_weight(self):
        # ConvTranspose2d weight: [IC, OC, K, K]
        w = self.conv_transpose.weight  # [IC, OC, K, K]
        # Flip spatial dims and transpose IC/OC
        w_flipped = torch.flip(w, dims=[2, 3])  # [IC, OC, K, K]
        w_conv = w_flipped.permute(1, 0, 2, 3).contiguous()  # [OC, IC, K, K]
        return w_conv

    def forward(self, x):
        bn = self.batch_norm
        if bn.training or self.stride != 1:
            # Fallback
            x = self.conv_transpose(x)
            x = bn(x)
            x = torch.tanh(x)
            x = self.max_pool(x)
            x = self.group_norm(x)
            return x

        N, IC, IH, IW = x.shape
        K = self.kernel_size
        # For stride=1, ConvTranspose2d output: OH = IH + K - 1 - 2*padding
        # Equivalent Conv2d with weight flipped and padding = K - 1 - padding
        PAD = K - 1 - self.padding
        OH = IH + K - 1 - 2 * self.padding
        OW = IW + K - 1 - 2 * self.padding
        OC = self.out_channels

        # Get fused BN scale/shift (incorporating conv_transpose bias)
        # ConvTranspose2d has bias [OC]
        conv_bias = self.conv_transpose.bias  # [OC]
        bn_scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)  # [OC]
        bn_shift = bn.bias - bn.running_mean * bn_scale  # [OC]
        # After conv: y = conv + conv_bias
        # bn(y) = (y - mean) * (w / sqrt(v+e)) + b = y * scale + (shift)
        # where shift_total = (conv_bias)*scale + shift
        fused_scale = bn_scale
        fused_shift = conv_bias * bn_scale + bn_shift

        # Get equivalent conv weight
        w_conv = self._get_conv_weight()

        x = x.contiguous()
        conv_out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 64
        BLOCK_SP = 64
        IC_BLOCK = 32

        # Ensure BLOCK_OC is at least OC if OC is small
        if OC < BLOCK_OC:
            BLOCK_OC = OC
        if IC < IC_BLOCK:
            # Find power of 2 <= IC
            IC_BLOCK = 1
            while IC_BLOCK * 2 <= IC:
                IC_BLOCK *= 2

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_SP))
        conv_bn_tanh_kernel[grid](
            x, w_conv, fused_scale.contiguous(), fused_shift.contiguous(), conv_out,
            N, IC, IH, IW,
            OC, OH, OW,
            K=K, PAD=PAD,
            BLOCK_OC=BLOCK_OC,
            BLOCK_SP=BLOCK_SP,
            IC_BLOCK=IC_BLOCK,
            num_warps=4,
            num_stages=2,
        )

        # Pool + GN
        H_out = OH // 2
        W_out = OW // 2
        G = self.num_groups
        C = OC
        CPG = C // G
        HW_OUT = H_out * W_out

        out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

        BLOCK = 1
        while BLOCK < HW_OUT:
            BLOCK *= 2
        if BLOCK < 16:
            BLOCK = 16

        grid2 = (N * G,)
        pool_gn_kernel[grid2](
            conv_out, out,
            self.group_norm.weight.contiguous(), self.group_norm.bias.contiguous(),
            N, C, OH, OW,
            H_out, W_out,
            G,
            self.group_norm.eps,
            CPG=CPG,
            HW_OUT=HW_OUT,
            BLOCK=BLOCK,
            num_warps=4,
            num_stages=2,
        )
        return out