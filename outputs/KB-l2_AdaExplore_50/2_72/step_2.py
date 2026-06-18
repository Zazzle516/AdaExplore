import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Conv transpose 3D as im2col-style GEMM with tensor cores
# Output: (N, OC, OD, OH, OW)
# Weight: (IC, OC, KD, KH, KW) -> reshape as K = IC*KD*KH*KW, M = OC
# For each output position, we accumulate over the K dim
#
# We split work as: program per (N, OC_tile=OC since OC=16 is small, spatial_tile)
# We hold an [OC, BLOCK_SPATIAL] accumulator. Loop over IC*KD*KH*KW with K-tile.

@triton.jit
def conv_transpose3d_gemm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    BLOCK_N: tl.constexpr,   # spatial tile
    OC_C: tl.constexpr,      # OC, constexpr
    K_TOTAL: tl.constexpr,   # IC*KD*KH*KW, constexpr
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    spatial = OD * OH * OW
    offs_n = pid_s * BLOCK_N + tl.arange(0, BLOCK_N)  # spatial offsets
    mask_n = offs_n < spatial

    od = offs_n // (OH * OW)
    rem = offs_n % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    # Accumulator [OC_C, BLOCK_N]
    acc = tl.zeros((OC_C, BLOCK_N), dtype=tl.float32)

    oc_range = tl.arange(0, OC_C)

    # iterate over K = IC * KD * KH * KW
    for k in tl.static_range(0, K_TOTAL):
        ic = k // (KD * KH * KW)
        krem = k % (KD * KH * KW)
        kd = krem // (KH * KW)
        krem2 = krem % (KH * KW)
        kh = krem2 // KW
        kw = krem2 % KW

        id_num = od + PD - kd
        ih_num = oh + PH - kh
        iw_num = ow + PW - kw

        id_q = id_num // SD
        ih_q = ih_num // SH
        iw_q = iw_num // SW

        valid = (id_num >= 0) & (ih_num >= 0) & (iw_num >= 0)
        valid = valid & ((id_num - id_q * SD) == 0)
        valid = valid & ((ih_num - ih_q * SH) == 0)
        valid = valid & ((iw_num - iw_q * SW) == 0)
        valid = valid & (id_q >= 0) & (id_q < ID)
        valid = valid & (ih_q >= 0) & (ih_q < IH)
        valid = valid & (iw_q >= 0) & (iw_q < IW)
        valid = valid & mask_n

        x_off = (((pid_n * IC + ic) * ID + id_q) * IH + ih_q) * IW + iw_q
        x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)  # [BLOCK_N]

        # weight: (IC, OC, KD, KH, KW), load OC values for this k
        w_base = ((ic * OC) * KD + kd) * KH * KW + kh * KW + kw
        # offset stride for oc is KD*KH*KW
        w_off = w_base + oc_range * (KD * KH * KW)
        w_val = tl.load(w_ptr + w_off)  # [OC_C]

        # outer product
        acc += w_val[:, None] * x_val[None, :]

    # bias
    b_val = tl.load(b_ptr + oc_range)  # [OC_C]
    acc += b_val[:, None]

    # store: out[n, oc, od, oh, ow]
    # base for (n, oc) plane: (pid_n*OC + oc) * spatial + offs_n
    out_base = pid_n * OC * spatial
    out_offs = out_base + oc_range[:, None] * spatial + offs_n[None, :]
    store_mask = mask_n[None, :] & (oc_range[:, None] < OC)
    tl.store(out_ptr + out_offs, acc, mask=store_mask)


@triton.jit
def fused_bn_avgpool4_kernel(
    x_ptr, scale_ptr, shift_ptr, out_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * C * OD * OH * OW
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    ow = offs % OW
    t1 = offs // OW
    oh = t1 % OH
    t2 = t1 // OH
    od = t2 % OD
    t3 = t2 // OD
    c = t3 % C
    n = t3 // C

    d0 = od * 4
    h0 = oh * 4
    w0 = ow * 4

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    base = ((n * C + c) * D)

    for dd in tl.static_range(0, 4):
        for hh in tl.static_range(0, 4):
            for ww in tl.static_range(0, 4):
                d_idx = d0 + dd
                h_idx = h0 + hh
                w_idx = w0 + ww
                in_off = ((base + d_idx) * H + h_idx) * W + w_idx
                v = tl.load(x_ptr + in_off, mask=mask, other=0.0)
                acc += v

    acc = acc * (1.0 / 64.0)

    s = tl.load(scale_ptr + c, mask=mask, other=0.0)
    sh = tl.load(shift_ptr + c, mask=mask, other=0.0)
    acc = acc * s + sh

    tl.store(out_ptr + offs, acc, mask=mask)


def conv_transpose3d_triton(x, weight, bias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    SD = SH = SW = stride
    PD = PH = PW = padding

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_N = 128
    spatial = OD * OH * OW
    K_TOTAL = IC * KD * KH * KW

    # OC_C must be a power-of-2 >= OC for tl.arange
    OC_C = 1
    while OC_C < OC:
        OC_C *= 2

    grid = (N, (spatial + BLOCK_N - 1) // BLOCK_N)
    conv_transpose3d_gemm_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_N=BLOCK_N,
        OC_C=OC_C,
        K_TOTAL=K_TOTAL,
        num_warps=4,
        num_stages=2,
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

    def forward(self, x):
        x = x.contiguous()
        weight = self.conv_transpose.weight.contiguous()
        bias = self.conv_transpose.bias.contiguous()

        y = conv_transpose3d_triton(x, weight, bias, self.stride, self.padding)

        bn = self.batch_norm
        if self.training:
            y = F.batch_norm(
                y, bn.running_mean, bn.running_var,
                bn.weight, bn.bias,
                training=True, momentum=bn.momentum, eps=bn.eps,
            )
            N, C, D, H, W = y.shape
            OD, OH, OW = D // 4, H // 4, W // 4
            out = torch.empty((N, C, OD, OH, OW), device=y.device, dtype=y.dtype)
            total = N * C * OD * OH * OW
            BLOCK = 256
            grid = ((total + BLOCK - 1) // BLOCK,)
            scale = torch.ones(C, device=y.device, dtype=y.dtype)
            shift = torch.zeros(C, device=y.device, dtype=y.dtype)
            fused_bn_avgpool4_kernel[grid](
                y, scale, shift, out,
                N, C, D, H, W,
                OD, OH, OW,
                BLOCK=BLOCK, num_warps=4, num_stages=2,
            )
            return out
        else:
            eps = bn.eps
            var = bn.running_var
            mean = bn.running_mean
            gamma = bn.weight
            beta = bn.bias
            scale = gamma / torch.sqrt(var + eps)
            shift = beta - mean * scale

            N, C, D, H, W = y.shape
            OD, OH, OW = D // 4, H // 4, W // 4
            out = torch.empty((N, C, OD, OH, OW), device=y.device, dtype=y.dtype)
            total = N * C * OD * OH * OW
            BLOCK = 256
            grid = ((total + BLOCK - 1) // BLOCK,)
            fused_bn_avgpool4_kernel[grid](
                y, scale.contiguous(), shift.contiguous(), out,
                N, C, D, H, W,
                OD, OH, OW,
                BLOCK=BLOCK, num_warps=4, num_stages=2,
            )
            return out