import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def scatter_conv_t3d_kernel(
    x_ptr,        # [N, IC, D, H, W]
    w_ptr,        # [IC, OC, KD, KH, KW]
    acc_ptr,      # [N, OC, OD, OH, OW] float32 accumulator
    N, IC, OC,
    D, H, W,
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    stride_d: tl.constexpr, stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_d: tl.constexpr, pad_h: tl.constexpr, pad_w: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    OC_C: tl.constexpr,
    IC_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)   # input d position
    pid_hw = tl.program_id(2)  # tile of (h, w) flat

    # Decode (h, w)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offs < (H * W)
    ih = hw_offs // W
    iw = hw_offs % W
    id_ = pid_d

    # Load input slice [IC, BLOCK_HW] for this (n, id, ih, iw)
    ic_offs = tl.arange(0, IC_C)
    ic_mask = ic_offs < IC

    # x[n, ic, id, ih, iw]
    x_base = pid_n * (IC * D * H * W) + id_ * (H * W)
    x_ptrs = x_ptr + x_base + ic_offs[:, None] * (D * H * W) + (ih * W + iw)[None, :]
    x_vals = tl.load(x_ptrs, mask=ic_mask[:, None] & hw_mask[None, :], other=0.0)  # [IC_C, BLOCK_HW]

    oc_offs = tl.arange(0, OC_C)
    oc_mask = oc_offs < OC

    # Loop over kernel positions
    for kd in tl.static_range(0, KD):
        od = id_ * stride_d + kd - pad_d
        od_valid = (od >= 0) & (od < OD)
        for kh in tl.static_range(0, KH):
            oh = ih * stride_h + kh - pad_h
            oh_valid = (oh >= 0) & (oh < OH)
            for kw in tl.static_range(0, KW):
                ow = iw * stride_w + kw - pad_w
                ow_valid = (ow >= 0) & (ow < OW)

                spatial_valid = oh_valid & ow_valid & hw_mask  # [BLOCK_HW]

                if od_valid:
                    # Load weight slice [IC, OC] for this (kd, kh, kw)
                    w_ptrs = w_ptr + ic_offs[:, None] * (OC * KD * KH * KW) + oc_offs[None, :] * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w_vals = tl.load(w_ptrs, mask=ic_mask[:, None] & oc_mask[None, :], other=0.0)  # [IC_C, OC_C]

                    # outer product over IC: x_vals.T @ w_vals -> [BLOCK_HW, OC]
                    # x_vals: [IC_C, BLOCK_HW], w_vals: [IC_C, OC_C]
                    contrib = tl.sum(x_vals[:, :, None] * w_vals[:, None, :], axis=0)  # [BLOCK_HW, OC_C]

                    # scatter-add to acc[n, oc, od, oh, ow]
                    out_spatial = od * (OH * OW) + oh * OW + ow  # [BLOCK_HW]
                    out_ptrs = acc_ptr + pid_n * (OC * OD * OH * OW) + oc_offs[None, :] * (OD * OH * OW) + out_spatial[:, None]
                    tl.atomic_add(out_ptrs, contrib, mask=spatial_valid[:, None] & oc_mask[None, :])


@triton.jit
def fused_epilogue_kernel(
    acc_ptr,    # [N, OC, OD, OH, OW]
    cbias_ptr,  # [OC]
    sbias_ptr,  # scalar
    out_ptr,    # [N, 1, OD, OH, OW]
    SPATIAL,
    NSPATIAL,
    OC: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    s_offs = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < NSPATIAL

    n = s_offs // SPATIAL
    sp = s_offs % SPATIAL

    base = n * (OC * SPATIAL) + sp
    c_offs = tl.arange(0, OC)
    ptrs = acc_ptr + base[:, None] + c_offs[None, :] * SPATIAL
    vals = tl.load(ptrs, mask=s_mask[:, None], other=-float('inf'))

    cb = tl.load(cbias_ptr + c_offs)
    vals = vals + cb[None, :]

    m = tl.max(vals, axis=1)
    e = tl.exp(vals - m[:, None])
    s = tl.sum(e, axis=1)
    lse = m + tl.log(s)

    sig = 1.0 / (1.0 + tl.exp(-(lse + 3.0)))
    hs = lse * sig / 6.0

    b = tl.load(sbias_ptr)
    y = hs - b
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)

    tl.store(out_ptr + s_offs, y, mask=s_mask)


def conv_transpose3d_fused(x, weight, conv_bias, scalar_bias, stride, padding, kernel_size):
    N, IC, D, H, W = x.shape
    OC = weight.shape[1]
    KD = KH = KW = kernel_size
    OD = (D - 1) * stride - 2 * padding + KD
    OH = (H - 1) * stride - 2 * padding + KH
    OW = (W - 1) * stride - 2 * padding + KW

    acc = torch.zeros((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

    BLOCK_HW = 128
    OC_C = triton.next_power_of_2(OC)
    if OC_C < 16:
        OC_C = 16
    IC_C = triton.next_power_of_2(IC)
    if IC_C < 4:
        IC_C = 4

    grid = (N, D, triton.cdiv(H * W, BLOCK_HW))
    scatter_conv_t3d_kernel[grid](
        x, weight, acc,
        N, IC, OC,
        D, H, W,
        OD, OH, OW,
        KD=KD, KH=KH, KW=KW,
        stride_d=stride, stride_h=stride, stride_w=stride,
        pad_d=padding, pad_h=padding, pad_w=padding,
        BLOCK_HW=BLOCK_HW,
        OC_C=OC_C,
        IC_C=IC_C,
        num_warps=4,
        num_stages=2,
    )

    out = torch.empty((N, 1, OD, OH, OW), device=x.device, dtype=torch.float32)
    SPATIAL = OD * OH * OW
    NSPATIAL = N * SPATIAL
    BLOCK_S = 512
    grid2 = (triton.cdiv(NSPATIAL, BLOCK_S),)
    fused_epilogue_kernel[grid2](
        acc, conv_bias, scalar_bias, out,
        SPATIAL, NSPATIAL,
        OC=OC, BLOCK_S=BLOCK_S,
        num_warps=4, num_stages=2,
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
        w = self.conv_transpose.weight.contiguous()
        cb = self.conv_transpose.bias.contiguous()
        sb = self.bias.reshape(-1).contiguous()
        return conv_transpose3d_fused(x, w, cb, sb, self.stride, self.padding, self.kernel_size)