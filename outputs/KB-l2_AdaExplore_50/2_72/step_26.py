import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Fused conv-transpose 3d + BN affine + avgpool(4x4x4)
# One program computes a tile of pooled output voxels for a given (n, oc).
# For each pooled output voxel (od_p, oh_p, ow_p), it accumulates the 64
# corresponding conv-transpose outputs at positions (od_p*4+dd, oh_p*4+hh, ow_p*4+ww).
# Each conv-transpose output sums over (ic, kd, kh, kw) where input index is
# (od + PD - kd) / SD if divisible. We loop over (ic, kd, kh, kw) and accumulate
# the contribution into all 64 pooled positions simultaneously.

@triton.jit
def _fused_convt_bn_pool_kernel(
    x_ptr, w_ptr, bias_ptr, scale_ptr, shift_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    PD_OD, PD_OH, PD_OW,  # pooled output dims
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_noc = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_hw = tl.program_id(2)

    n = pid_noc // OC
    oc = pid_noc % OC

    nh_tiles = (PD_OH + BLOCK_H - 1) // BLOCK_H
    pid_h = pid_hw // ((PD_OW + BLOCK_W - 1) // BLOCK_W)
    pid_w = pid_hw % ((PD_OW + BLOCK_W - 1) // BLOCK_W)

    od_p = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # [BD]
    oh_p = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)  # [BH]
    ow_p = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)  # [BW]

    d_mask = od_p < PD_OD
    h_mask = oh_p < PD_OH
    w_mask = ow_p < PD_OW

    # accumulator: [BD, BH, BW]
    acc = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)

    # For each conv-output position in the 4^3 window we need:
    # od = od_p*4 + dd, valid input id = (od + PD - kd) / SD if divisible
    # We iterate kd, kh, kw, ic and accumulate.

    # Precompute the 4 sub-positions for each pooled coord:
    # od[bd, dd] = od_p[bd]*4 + dd
    # We'll fully unroll dd, hh, ww (64 iterations) inside kd/kh/kw loops.

    for kd in tl.static_range(0, KD):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                # For each of the 4 sub-positions per pooled coord,
                # compute input indices and validity, then accumulate.
                # We do this dd/hh/ww loop unrolled.
                for dd in tl.static_range(0, 4):
                    od = od_p * 4 + dd  # [BD]
                    num_d = od + PD - kd
                    id_valid_d = (num_d >= 0) & (num_d < ID * SD) & ((num_d % SD) == 0)
                    id_ = num_d // SD  # [BD]
                    id_valid_d = id_valid_d & (id_ >= 0) & (id_ < ID)
                    for hh in tl.static_range(0, 4):
                        oh = oh_p * 4 + hh  # [BH]
                        num_h = oh + PH - kh
                        id_valid_h = (num_h >= 0) & (num_h < IH * SH) & ((num_h % SH) == 0)
                        ih_ = num_h // SH
                        id_valid_h = id_valid_h & (ih_ >= 0) & (ih_ < IH)
                        for ww in tl.static_range(0, 4):
                            ow = ow_p * 4 + ww
                            num_w = ow + PW - kw
                            id_valid_w = (num_w >= 0) & (num_w < IW * SW) & ((num_w % SW) == 0)
                            iw_ = num_w // SW
                            id_valid_w = id_valid_w & (iw_ >= 0) & (iw_ < IW)

                            # valid[BD,BH,BW]
                            valid = (id_valid_d[:, None, None] &
                                     id_valid_h[None, :, None] &
                                     id_valid_w[None, None, :])

                            # Sum over IC of x[n, ic, id_, ih_, iw_] * w[ic, oc, kd, kh, kw]
                            # Loop over IC (small, e.g. 3)
                            partial = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)
                            for ic in range(0, IC):
                                # x offset: ((n*IC+ic)*ID + id_)*IH*IW + ih_*IW + iw_
                                # Build 3D offset
                                x_off = ((n * IC + ic) * ID + id_[:, None, None]) * (IH * IW) \
                                        + ih_[None, :, None] * IW + iw_[None, None, :]
                                x_v = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                                w_off = ((ic * OC + oc) * KD + kd) * KH * KW + kh * KW + kw
                                w_v = tl.load(w_ptr + w_off)
                                partial += x_v * w_v
                            acc += partial

    # Apply bias (conv bias) + BN affine
    bias_v = tl.load(bias_ptr + oc)
    scale_v = tl.load(scale_ptr + oc)
    shift_v = tl.load(shift_ptr + oc)

    # Each pooled voxel = avg of 64 conv-out values; conv-out = sum + bias.
    # bias is added 64 times then divided by 64 -> equals bias.
    # Similarly BN: out = (conv_val) * scale + shift, averaged.
    # avg = mean(conv_val) * scale + shift  (since scale/shift constant over window)
    # where conv_val_avg = acc/64 + bias
    conv_avg = acc / 64.0 + bias_v
    res = conv_avg * scale_v + shift_v

    # store
    out_off = (((n * OC + oc) * PD_OD + od_p[:, None, None]) * PD_OH + oh_p[None, :, None]) * PD_OW + ow_p[None, None, :]
    out_mask = d_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]
    tl.store(out_ptr + out_off, res, mask=out_mask)


def fused_convt_bn_pool(x, weight, bias, scale, shift, stride, padding):
    N, IC, ID, IH, IW = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    SD = SH = SW = stride
    PD = PH = PW = padding
    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    PD_OD = OD // 4
    PD_OH = OH // 4
    PD_OW = OW // 4

    out = torch.empty((N, OC, PD_OD, PD_OH, PD_OW), device=x.device, dtype=torch.float32)

    BLOCK_D = 2
    BLOCK_H = 4
    BLOCK_W = 8

    grid = (
        N * OC,
        (PD_OD + BLOCK_D - 1) // BLOCK_D,
        ((PD_OH + BLOCK_H - 1) // BLOCK_H) * ((PD_OW + BLOCK_W - 1) // BLOCK_W),
    )

    _fused_convt_bn_pool_kernel[grid](
        x, weight, bias, scale, shift, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        PD_OD, PD_OH, PD_OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_D=BLOCK_D, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous()
        bias = self.conv_transpose.bias
        if bias is None:
            bias = torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)
        else:
            bias = bias.contiguous()

        if self.training:
            # Fallback: run reference path
            y = self.conv_transpose(x)
            y = self.batch_norm(y)
            y = F.avg_pool3d(y, 2)
            y = F.avg_pool3d(y, 2)
            return y

        rm = self.batch_norm.running_mean
        rv = self.batch_norm.running_var
        eps = self.batch_norm.eps
        w = self.batch_norm.weight
        b = self.batch_norm.bias
        invstd = torch.rsqrt(rv + eps)
        scale = (w * invstd).contiguous()
        shift = (b - rm * scale).contiguous()

        return fused_convt_bn_pool(x, weight, bias, scale, shift, self.stride, self.padding)