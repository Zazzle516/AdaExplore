import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Fused ConvTranspose3d + BN affine + AvgPool(4x4x4)
# Each program computes BLOCK_PW pooled-W positions for a single (n, oc, pd, ph) row.
# OC=16 is small, so we parallelize across (n, oc, pd, ph, pw_block).
# We hoist the IC*KD*KH*KW weights for this OC into registers once.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_PW': 4}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_PW': 8}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_PW': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_PW': 16}, num_warps=4, num_stages=2),
    ],
    key=['PD_OD', 'PD_OH', 'PD_OW', 'IC'],
)
@triton.jit
def _fused_kernel(
    x_ptr, w_ptr, bias_ptr, scale_ptr, shift_ptr, out_ptr,
    N, IC: tl.constexpr, ID, IH, IW,
    OC: tl.constexpr, OD, OH, OW,
    PD_OD, PD_OH, PD_OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_PW: tl.constexpr,
):
    pid = tl.program_id(0)
    # decode (n, oc, pd, ph, pw_blk)
    nblocks_pw = (PD_OW + BLOCK_PW - 1) // BLOCK_PW
    pw_blk = pid % nblocks_pw
    t = pid // nblocks_pw
    ph = t % PD_OH
    t = t // PD_OH
    pd = t % PD_OD
    t = t // PD_OD
    oc = t % OC
    n = t // OC

    pw_offs = pw_blk * BLOCK_PW + tl.arange(0, BLOCK_PW)
    pw_mask = pw_offs < PD_OW

    acc = tl.zeros((BLOCK_PW,), dtype=tl.float32)

    # iterate over 4x4x4 pooled window
    for dd in tl.static_range(0, 4):
        od = pd * 4 + dd
        for kd in tl.static_range(0, KD):
            id_num = od + PD - kd
            id_q = id_num // SD
            id_r = id_num - id_q * SD
            d_ok = (id_r == 0) & (id_q >= 0) & (id_q < ID)
            for hh in tl.static_range(0, 4):
                oh = ph * 4 + hh
                for kh in tl.static_range(0, KH):
                    ih_num = oh + PH - kh
                    ih_q = ih_num // SH
                    ih_r = ih_num - ih_q * SH
                    h_ok = (ih_r == 0) & (ih_q >= 0) & (ih_q < IH)
                    dh_ok = d_ok & h_ok
                    if dh_ok:
                        for ww in tl.static_range(0, 4):
                            ow_vec = pw_offs * 4 + ww  # [BLOCK_PW]
                            for kw in tl.static_range(0, KW):
                                iw_num = ow_vec + PW - kw
                                iw_q = iw_num // SW
                                iw_r = iw_num - iw_q * SW
                                w_ok = (iw_r == 0) & (iw_q >= 0) & (iw_q < IW) & pw_mask
                                for ic in tl.static_range(0, IC):
                                    x_off = ((n * IC + ic) * ID + id_q) * IH * IW + ih_q * IW + iw_q
                                    x_val = tl.load(x_ptr + x_off, mask=w_ok, other=0.0)
                                    w_off = ((ic * OC + oc) * KD + kd) * KH * KW + kh * KW + kw
                                    w_val = tl.load(w_ptr + w_off)
                                    acc += x_val * w_val

    bias_val = tl.load(bias_ptr + oc)
    scale_val = tl.load(scale_ptr + oc)
    shift_val = tl.load(shift_ptr + oc)

    mean_v = (acc + 64.0 * bias_val) / 64.0
    result = mean_v * scale_val + shift_val

    out_off = ((n * OC + oc) * PD_OD + pd) * PD_OH * PD_OW + ph * PD_OW + pw_offs
    tl.store(out_ptr + out_off, result, mask=pw_mask)


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

    def grid(meta):
        nblocks_pw = (PD_OW + meta['BLOCK_PW'] - 1) // meta['BLOCK_PW']
        return (N * OC * PD_OD * PD_OH * nblocks_pw,)

    _fused_kernel[grid](
        x.contiguous(), weight.contiguous(), bias.contiguous(),
        scale.contiguous(), shift.contiguous(), out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        PD_OD, PD_OH, PD_OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
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
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias
        if bias is None:
            bias = torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)

        if self.training:
            y = self.conv_transpose(x)
            y = self.batch_norm(y)
            y = F.avg_pool3d(y, 2)
            y = F.avg_pool3d(y, 2)
            return y

        rm = self.batch_norm.running_mean
        rv = self.batch_norm.running_var
        eps = self.batch_norm.eps
        bn_w = self.batch_norm.weight
        bn_b = self.batch_norm.bias
        invstd = torch.rsqrt(rv + eps)
        scale = bn_w * invstd
        shift = bn_b - rm * scale

        return fused_convt_bn_pool(x, weight, bias, scale, shift, self.stride, self.padding)