import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_fused_kernel(
    x_ptr,           # [N, IC, D, H, W]
    w_ptr,           # [IC, OC, KD, KH, KW]
    b_ptr,           # [OC]
    bias_ptr,        # scalar bias
    out_ptr,         # [N, 1, OD, OH, OW]
    N, IC, OC,
    D, H, W,
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    stride_d: tl.constexpr, stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_d: tl.constexpr, pad_h: tl.constexpr, pad_w: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
    OC_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    # Spatial tile: BLOCK_SPATIAL output voxels along the flat (OD*OH*OW) dim
    s_offs = pid_s * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    s_mask = s_offs < (OD * OH * OW)

    # Decompose flat index -> (od, oh, ow)
    od = s_offs // (OH * OW)
    rem = s_offs % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    # Accumulator [BLOCK_SPATIAL, OC]
    acc = tl.zeros((BLOCK_SPATIAL, OC_C), dtype=tl.float32)

    oc_offs = tl.arange(0, OC_C)
    oc_mask = oc_offs < OC

    # For each output voxel (od,oh,ow), find input positions (id,ih,iw) and kernel offsets (kd,kh,kw):
    #   od = id*stride_d + kd - pad_d  =>  kd = od + pad_d - id*stride_d
    # We iterate over (kd, kh, kw) and compute the required (id, ih, iw).
    for kd in tl.static_range(0, KD):
        # id*stride_d = od + pad_d - kd
        num_d = od + pad_d - kd
        id_ = num_d // stride_d
        id_valid = (num_d - id_ * stride_d == 0) & (id_ >= 0) & (id_ < D)
        for kh in tl.static_range(0, KH):
            num_h = oh + pad_h - kh
            ih_ = num_h // stride_h
            ih_valid = (num_h - ih_ * stride_h == 0) & (ih_ >= 0) & (ih_ < H)
            for kw in tl.static_range(0, KW):
                num_w = ow + pad_w - kw
                iw_ = num_w // stride_w
                iw_valid = (num_w - iw_ * stride_w == 0) & (iw_ >= 0) & (iw_ < W)

                valid = id_valid & ih_valid & iw_valid & s_mask  # [BLOCK_SPATIAL]

                # Compute input index for each spatial position
                # x layout [N, IC, D, H, W]
                # For each IC, gather x[n, ic, id, ih, iw] and multiply by w[ic, :, kd, kh, kw]
                spatial_in_off = id_ * (H * W) + ih_ * W + iw_  # [BLOCK_SPATIAL]

                for ic in range(0, IC):
                    x_off = pid_n * (IC * D * H * W) + ic * (D * H * W) + spatial_in_off
                    x_vals = tl.load(x_ptr + x_off, mask=valid, other=0.0)  # [BLOCK_SPATIAL]

                    # weight w[ic, oc, kd, kh, kw], OC contiguous in oc dim
                    w_off = ic * (OC * KD * KH * KW) + oc_offs * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w_vals = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [OC_C]

                    acc += x_vals[:, None] * w_vals[None, :]

    # Add conv bias (per-OC)
    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + b_vals[None, :]

    # mask invalid OC channels with -inf for logsumexp
    neg_inf = float('-inf')
    acc = tl.where(oc_mask[None, :], acc, neg_inf)

    # logsumexp over OC
    m = tl.max(acc, axis=1)  # [BLOCK_SPATIAL]
    e = tl.exp(acc - m[:, None])
    s_sum = tl.sum(e, axis=1)
    lse = m + tl.log(s_sum)

    # HardSwish: x * sigmoid(x+3) / 6
    sig = 1.0 / (1.0 + tl.exp(-(lse + 3.0)))
    hs = lse * sig / 6.0

    # subtract scalar bias
    b_scalar = tl.load(bias_ptr)
    y = hs - b_scalar

    # clamp [-1, 1]
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)

    # store: out [N, 1, OD, OH, OW]
    out_off = pid_n * (OD * OH * OW) + s_offs
    tl.store(out_ptr + out_off, y, mask=s_mask)


def conv_transpose3d_fused(x, weight, conv_bias, scalar_bias,
                            stride, padding, kernel_size):
    N, IC, D, H, W = x.shape
    OC = weight.shape[1]
    KD = KH = KW = kernel_size
    OD = (D - 1) * stride - 2 * padding + KD
    OH = (H - 1) * stride - 2 * padding + KH
    OW = (W - 1) * stride - 2 * padding + KW

    out = torch.empty((N, 1, OD, OH, OW), device=x.device, dtype=torch.float32)

    BLOCK_SPATIAL = 64
    OC_C = triton.next_power_of_2(OC)
    if OC_C < 16:
        OC_C = 16

    spatial_total = OD * OH * OW
    grid = (N, triton.cdiv(spatial_total, BLOCK_SPATIAL))

    conv_transpose3d_fused_kernel[grid](
        x, weight, conv_bias, scalar_bias, out,
        N, IC, OC,
        D, H, W,
        OD, OH, OW,
        KD=KD, KH=KH, KW=KW,
        stride_d=stride, stride_h=stride, stride_w=stride,
        pad_d=padding, pad_h=padding, pad_w=padding,
        BLOCK_SPATIAL=BLOCK_SPATIAL,
        OC_C=OC_C,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x):
        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()  # [IC, OC, KD, KH, KW]
        cb = self.conv_transpose.bias.contiguous()   # [OC]
        sb = self.bias.reshape(-1).contiguous()      # scalar
        return conv_transpose3d_fused(
            x, w, cb, sb,
            self.stride, self.padding, self.kernel_size
        )