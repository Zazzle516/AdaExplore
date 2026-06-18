import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_mean_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, D, H, W,
    OC, KD, KH, KW,
    PAD_D, PAD_H, PAD_W,
    OD, OH, OW,
    BLOCK_HW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, oc_tile, hw_tile)
    # accumulates the full conv output for this tile across all output depth,
    # summing into a single (BLOCK_OC, BLOCK_HW) register tile, then /OD.
    pid = tl.program_id(0)
    n = tl.program_id(1)
    oc_tile = tl.program_id(2)

    num_hw_tiles = tl.cdiv(OH * OW, BLOCK_HW)
    hw_tile = pid

    offs_hw = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = oc_tile * BLOCK_OC + tl.arange(0, BLOCK_OC)

    oh = offs_hw // OW
    ow = offs_hw % OW
    hw_mask = offs_hw < (OH * OW)
    oc_mask = offs_oc < OC

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    # For ConvTranspose3d with stride=1: equivalent to Conv3d with flipped kernel
    # and padding=KD-1-PAD. We implement as direct conv:
    # out[n,oc,od,oh,ow] = sum_{ic,kd,kh,kw} x[n,ic, od+kd-PAD', oh+kh-PAD'', ow+kw-PAD'''] * w_flipped[oc,ic,kd,kh,kw]
    # Equivalently for ConvTranspose with weight w[ic,oc,kd,kh,kw]:
    # out[n,oc,od,oh,ow] = sum_{ic,kd,kh,kw} x[n,ic, od-kd+PAD_eff, ...] * w[ic,oc,kd,kh,kw]
    # where the input position is (od + (KD-1) - kd - PAD) for ConvT with stride=1.
    # We'll just use ConvT formula directly.

    # Sum over output depth od in [0, OD)
    # acc[oc,hw] = sum_{od} sum_{ic,kd,kh,kw} x[n,ic, od+(KD-1)-kd-PAD_D, oh+(KH-1)-kh-PAD_H, ow+(KW-1)-kw-PAD_W] * w[ic,oc,kd,kh,kw]
    # Rearrange: let id = od - kd + (KD-1) - PAD_D. As od ranges over [0,OD) and kd over [0,KD),
    # we can swap order: for each ic, kd, kh, kw, sum x over od for valid id.
    # But easier: for each kd, kh, kw, the input depth index is id = od + offset_d(kd).
    # Sum over od of x[n,ic,id,ih,iw] for od in [0,OD) where id in [0,D).
    # id = od + (KD-1-kd-PAD_D). So od_min = max(0, -off_d), od_max = min(OD, D - off_d).
    # The depth-sum of x[n,ic,id,ih,iw] over valid od equals sum over valid id.
    # Hmm, but ih, iw depend on (kh,kw) and oh,ow individually for each hw element.

    # Strategy: loop ic, kd, kh, kw. Compute input h,w indices per hw element.
    # Compute the depth-sum S_ic_kd[n,ic,kd,ih,iw] = sum_{od valid} x[n,ic, od+off_d, ih, iw]
    # This is a sum over a contiguous range of depth indices [id_lo, id_hi).
    # We do this on the fly via a loop over id. To avoid that inner loop, we can
    # precompute cumulative sum of x over D... but we cannot rewrite weights.
    # Instead loop over id in [0, D) inside the kernel? That's D=32 iterations × IC*KD*KH*KW = 16*27*32 ≈ 13824. Too many.

    # Better: swap the loops. Outer: ic, kh, kw. Compute ih,iw masks. For each (kd):
    # determine valid od range, which determines valid id range. The depth sum of x
    # over that id range gives the contribution. We loop kd inside ic,kh,kw.
    # For each (ic, kh, kw), we need sum of x[n,ic,id_range,ih,iw] for 3 different ranges (one per kd).
    # Total inner loops: IC * KH * KW = 16*9 = 144, each doing 3 partial sums by iterating id.
    # The id ranges overlap heavily (each covers ~OD elements). Better:
    # Loop ic, kh, kw. Compute the full depth sum once: T = sum_{id=0}^{D-1} x[n,ic,id,ih,iw].
    # Then for each kd, the valid od range gives an id range. Since stride=1 and KD=3, PAD_D=1:
    # off_d(kd) = (KD-1) - kd - PAD_D = 2 - kd - 1 = 1 - kd. So for kd=0: off=1, id=od+1, od in [0,OD), id in [1,OD+1). Valid id in [1, min(OD+1,D)).
    # For kd=1: off=0, id=od, id in [0,OD)∩[0,D) = [0, min(OD,D)) = [0,32).
    # For kd=2: off=-1, id=od-1, id in [-1,OD-1)∩[0,D) = [0, OD-1) = [0,31).
    # 
    # OD for ConvT3d stride=1: OD = D - 2*PAD_D + KD - 1 + output_padding = 32 - 2 + 2 = 32.
    # So id_kd0_range = [1, 32), id_kd1_range = [0, 32), id_kd2_range = [0, 31).
    #
    # We can compute three partial sums in one pass over id:
    # S0 = sum id in [1,32) x[id]    (= T - x[0])
    # S1 = sum id in [0,32) x[id]   = T
    # S2 = sum id in [0,31) x[id]   = T - x[31]
    # 
    # Great — only need T and x[0], x[31] per (ic,kh,kw,ih,iw)!
    # 
    # But this is the safety-contract violation: precomputing depth-sums and using
    # them in a smaller GEMM-like reduction. The contract forbids "pre-reducing
    # along any axis that a downstream linear reduction will later collapse".
    #
    # Therefore: we must materialize the full depth dimension in the conv output.
    # We loop over od explicitly.

    for od in range(0, OD):
        # For each od, accumulate conv output[n, oc_tile, od, hw_tile] into a temp,
        # then add to acc (which sums over od and will be divided by OD).
        tmp = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)
        for ic in range(0, IC):
            for kd in range(0, KD):
                id = od + (KD - 1) - kd - PAD_D
                if (id >= 0) and (id < D):
                    for kh in range(0, KH):
                        ih = oh + (KH - 1) - kh - PAD_H
                        ih_mask = (ih >= 0) & (ih < H)
                        for kw in range(0, KW):
                            iw = ow + (KW - 1) - kw - PAD_W
                            iw_mask = (iw >= 0) & (iw < W)
                            in_mask = hw_mask & ih_mask & iw_mask
                            x_off = ((n * IC + ic) * D + id) * H * W + ih * W + iw
                            x_val = tl.load(x_ptr + x_off, mask=in_mask, other=0.0)  # (BLOCK_HW,)
                            w_off = ((ic * OC + offs_oc) * KD + kd) * KH * KW + kh * KW + kw
                            w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # (BLOCK_OC,)
                            tmp += w_val[:, None] * x_val[None, :]
        acc += tmp

    # divide by OD for mean
    acc = acc / OD
    # add conv bias (already broadcast over spatial) — we fold conv bias here
    bias_val = tl.load(b_ptr + offs_oc, mask=oc_mask, other=0.0)
    acc += bias_val[:, None]

    # store to out[n, oc, 0, oh, ow]
    out_off = (n * OC + offs_oc[:, None]) * (OH * OW) + offs_hw[None, :]
    store_mask = oc_mask[:, None] & hw_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=store_mask)


@triton.jit
def softmax_tanh_scale_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, HW,
    scaling_factor,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // HW
    hw = pid % HW

    offs = tl.arange(0, BLOCK_C)
    mask = offs < C

    base = n * C * HW + offs * HW + hw
    x = tl.load(x_ptr + base, mask=mask, other=-float('inf'))
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
    x = x + b

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s
    y = tl.extra.cuda.libdevice.tanh(sm) * scaling_factor

    tl.store(out_ptr + base, y, mask=mask)


def fused_softmax_tanh_scale(x: torch.Tensor, bias: torch.Tensor, scaling_factor: float):
    B, C, D, H, W = x.shape
    assert D == 1
    x_view = x.view(B, C, H * W).contiguous()
    bias_view = bias.view(-1).contiguous()
    out = torch.empty_like(x_view)
    HW = H * W
    BLOCK_C = triton.next_power_of_2(C)
    grid = (B * HW,)
    softmax_tanh_scale_kernel[grid](
        x_view, bias_view, out, B, C, HW, scaling_factor,
        BLOCK_C=BLOCK_C, num_warps=2,
    )
    return out.view(B, C, 1, H, W)


def conv_transpose_mean(x, weight, bias, padding):
    # x: (N, IC, D, H, W), weight: (IC, OC, KD, KH, KW)
    N, IC, D, H, W = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    assert IC == IC_w
    PAD_D = padding
    PAD_H = padding
    PAD_W = padding
    # ConvTranspose3d output size with stride=1, output_padding=0:
    OD = D - 2 * PAD_D + KD - 1
    OH = H - 2 * PAD_H + KH - 1
    OW = W - 2 * PAD_W + KW - 1

    out = torch.empty((N, OC, 1, OH, OW), device=x.device, dtype=x.dtype)
    out_flat = out.view(N, OC, OH * OW)

    BLOCK_HW = 64
    BLOCK_OC = 32
    num_hw_tiles = triton.cdiv(OH * OW, BLOCK_HW)
    num_oc_tiles = triton.cdiv(OC, BLOCK_OC)

    grid = (num_hw_tiles, N, num_oc_tiles)
    conv_mean_kernel[grid](
        x, weight, bias, out_flat,
        N, IC, D, H, W,
        OC, KD, KH, KW,
        PAD_D, PAD_H, PAD_W,
        OD, OH, OW,
        BLOCK_HW=BLOCK_HW, BLOCK_OC=BLOCK_OC,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        if self.stride == 1:
            w = self.conv_transpose.weight
            cb = self.conv_transpose.bias
            x = x.contiguous()
            y = conv_transpose_mean(x, w, cb, self.padding)
            y = fused_softmax_tanh_scale(y, self.bias, self.scaling_factor)
            return y
        else:
            x = self.conv_transpose(x)
            x = x.mean(dim=2, keepdim=True)
            x = x + self.bias
            x = fused_softmax_tanh_scale(x.contiguous(), torch.zeros_like(self.bias).view(-1), self.scaling_factor)
            return x