import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ConvTranspose3d as a gather:
# y[n, oc, od, oh, ow] = sum_{ic, kd, kh, kw}
#     x[n, ic, id, ih, iw] * w[ic, oc, kd, kh, kw]
# where id*stride = od + pad - kd, etc., with divisibility check.
#
# Layout: input/output in NCDHW contiguous.
# We tile output over (N, OC_tile, spatial_tile).
# Inner reduction loops: kd, kh, kw, ic.

@triton.jit
def conv_transpose3d_gather_kernel(
    x_ptr,         # [N, IC, ID, IH, IW]
    w_ptr,         # [IC, OC, KD, KH, KW]
    b_ptr,         # [OC]
    out_ptr,       # [N, OC, OD, OH, OW]
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    MIN_VAL, INV_DIV,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)         # [BLOCK_OC]
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)         # [BLOCK_SP]

    oc_mask = oc_offs < OC                                       # [BLOCK_OC]
    sp_mask = sp_offs < (OD * OH * OW)                           # [BLOCK_SP]

    # decode od, oh, ow from sp_offs
    ow = sp_offs % OW
    tmp = sp_offs // OW
    oh = tmp % OH
    od = tmp // OH

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # base offsets
    n = pid_n
    x_base = n * IC * ID * IH * IW
    out_base = n * OC * OD * OH * OW

    # Iterate over kernel positions; compute id, ih, iw and validity
    for kd in tl.static_range(0, KD):
        id_num = od + PAD - kd
        id_ = id_num // STRIDE
        id_valid = (id_num % STRIDE == 0) & (id_ >= 0) & (id_ < ID)   # [BLOCK_SP]
        for kh in tl.static_range(0, KH):
            ih_num = oh + PAD - kh
            ih_ = ih_num // STRIDE
            ih_valid = (ih_num % STRIDE == 0) & (ih_ >= 0) & (ih_ < IH)
            for kw in tl.static_range(0, KW):
                iw_num = ow + PAD - kw
                iw_ = iw_num // STRIDE
                iw_valid = (iw_num % STRIDE == 0) & (iw_ >= 0) & (iw_ < IW)

                spatial_valid = id_valid & ih_valid & iw_valid & sp_mask  # [BLOCK_SP]

                # input spatial offset (per sp); same for all ic and oc
                in_sp_off = id_ * (IH * IW) + ih_ * IW + iw_              # [BLOCK_SP]

                # weight offset base for this kernel pos: w[ic, oc, kd, kh, kw]
                # stride: ic -> OC*KD*KH*KW, oc -> KD*KH*KW
                w_kpos = kd * (KH * KW) + kh * KW + kw

                # Reduction over IC
                for ic in range(0, IC):
                    # x value: [BLOCK_SP]
                    x_off = x_base + ic * (ID * IH * IW) + in_sp_off
                    x_val = tl.load(x_ptr + x_off, mask=spatial_valid, other=0.0)

                    # weight: [BLOCK_OC]
                    w_off = ic * (OC * KD * KH * KW) + oc_offs * (KD * KH * KW) + w_kpos
                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)

                    # outer product accumulate
                    acc += w_val[:, None] * x_val[None, :]

    # add bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)     # [BLOCK_OC]
    acc = acc + bias[:, None]

    # clamp + divide
    acc = tl.maximum(acc, MIN_VAL)
    acc = acc * INV_DIV

    # store
    out_off = out_base + oc_offs[:, None] * (OD * OH * OW) + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


def conv_transpose3d_fused(x, weight, bias, stride, padding, min_val, divisor):
    N, IC, ID, IH, IW = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    assert IC == IC_w

    OD = (ID - 1) * stride - 2 * padding + KD
    OH = (IH - 1) * stride - 2 * padding + KH
    OW = (IW - 1) * stride - 2 * padding + KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 32
    BLOCK_SP = 128

    grid = (
        N,
        triton.cdiv(OC, BLOCK_OC),
        triton.cdiv(OD * OH * OW, BLOCK_SP),
    )

    conv_transpose3d_gather_kernel[grid](
        x, weight, bias, out,
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        stride, padding,
        float(min_val), float(1.0 / divisor),
        BLOCK_OC=BLOCK_OC,
        BLOCK_SP=BLOCK_SP,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.stride = stride
        self.padding = padding
        self.min_value = min_value
        self.divisor = divisor

    def forward(self, x):
        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()
        b = self.conv_transpose.bias.contiguous()
        return conv_transpose3d_fused(x, w, b, self.stride, self.padding, self.min_value, self.divisor)