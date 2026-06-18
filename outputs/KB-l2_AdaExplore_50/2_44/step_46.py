import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Scatter-based ConvTranspose2d that fuses scalar multiply and per-(N,OC) sum reduction.
# Each program handles one (n, ic_tile, ih_tile, iw_tile) input region. It loads input,
# computes contribution to KH*KW*OC output positions per input pixel, and atomically adds
# the running sum (weighted by multiplier) into a (N, OC) buffer. We never materialize the
# full conv-transpose output.
#
# The contribution of input[n, ic, ih, iw] to output[n, oc, oh, ow] is
#   x[n,ic,ih,iw] * w[ic,oc,kh,kw]
# where oh = ih*stride - pad + kh, ow = iw*stride - pad + kw, with 0<=oh<OH, 0<=ow<OW.
# The sum over output spatial dims of contributions = (sum over valid kh,kw of w[ic,oc,kh,kw])
# times x[n,ic,ih,iw]. BUT this would violate the safety contract (pre-reducing weights).
# To preserve the heavy op's full multiply-add count, we must materialize the full output.
#
# So we do a gather-based im2col GEMM, fused with multiply + spatial-sum reduction.
# Each program computes a tile of the conv-transpose output, then reduces (sums) over
# the spatial tile and atomically adds into a (N, OC) buffer. Full MAC count preserved.


@triton.jit
def conv_transpose_fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    multiplier,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    oc_mask = oc_offs < OC
    hw_mask = hw_offs < (OH * OW)

    oh = hw_offs // OW
    ow = hw_offs % OW

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        ih_num = oh + PAD - kh
        ih = ih_num // STRIDE
        ih_valid = (ih_num % STRIDE == 0) & (ih >= 0) & (ih < IH)
        for kw in tl.static_range(0, KW):
            iw_num = ow + PAD - kw
            iw = iw_num // STRIDE
            iw_valid = (iw_num % STRIDE == 0) & (iw >= 0) & (iw < IW)
            valid = ih_valid & iw_valid & hw_mask  # [BLOCK_HW]

            for ic in range(0, IC):
                x_offset = pid_n * (IC * IH * IW) + ic * (IH * IW) + ih * IW + iw
                x_val = tl.load(x_ptr + x_offset, mask=valid, other=0.0)

                w_offset = ic * (OC * KH * KW) + oc_offs * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_offset, mask=oc_mask, other=0.0)

                acc += w_val[:, None] * x_val[None, :]

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    # acc is the conv-transpose output for this tile; add bias
    full = acc + bias[:, None]
    # mask invalid hw positions to zero before summing
    full = tl.where(hw_mask[None, :], full, 0.0)
    # multiply by scalar, then sum across spatial tile
    full = full * multiplier
    partial = tl.sum(full, axis=1)  # [BLOCK_OC]

    # atomic add into out[n, oc]
    out_offs = pid_n * OC + oc_offs
    tl.atomic_add(out_ptr + out_offs, partial, mask=oc_mask)


@triton.jit
def finalize_kernel(buf_ptr, out_ptr, total, inv_area, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    v = tl.load(buf_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, v * inv_area, mask=mask)


def fused_convtrans_mean(x, weight, bias, stride, padding, output_padding, multiplier):
    N, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w

    OH = (IH - 1) * stride - 2 * padding + KH + output_padding
    OW = (IW - 1) * stride - 2 * padding + KW + output_padding

    buf = torch.zeros((N, OC), device=x.device, dtype=torch.float32)

    BLOCK_OC = 32
    BLOCK_HW = 128

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_HW))

    conv_transpose_fused_kernel[grid](
        x, weight, bias, buf,
        N, IC, IH, IW,
        OC, OH, OW,
        multiplier,
        KH, KW,
        stride, padding,
        BLOCK_OC=BLOCK_OC,
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
        num_stages=2,
    )

    out = torch.empty((N, OC, 1, 1), device=x.device, dtype=x.dtype)
    inv_area = 1.0 / (OH * OW)
    total = N * OC
    BLOCK = 256
    finalize_kernel[(triton.cdiv(total, BLOCK),)](buf, out, total, inv_area, BLOCK=BLOCK)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.multiplier = multiplier
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous().cuda()
        bias = self.conv_transpose.bias.contiguous().cuda()

        out = fused_convtrans_mean(
            x, weight, bias,
            self.stride, self.padding, self.output_padding,
            self.multiplier,
        )
        return out