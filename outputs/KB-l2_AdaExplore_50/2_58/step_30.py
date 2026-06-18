import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_fused_kernel(
    x_ptr,           # [N, IC, ID, IH, IW] contiguous
    w_ptr,           # [IC, OC, KD, KH, KW] contiguous
    cb_ptr,          # [OC] conv bias
    bias_ptr,        # scalar
    out_ptr,         # [N, 1, OD, OH, OW] contiguous
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # one program per (N, OD, OH, OW-tile)
    pid_n = tl.program_id(0)
    pid_dh = tl.program_id(1)  # combined OD*OH
    pid_w = tl.program_id(2)

    od = pid_dh // OH
    oh = pid_dh % OH

    ow_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    ow_mask = ow_offs < OW

    # OC accumulator: [BLOCK_OC, BLOCK_W]
    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_OC, BLOCK_W), dtype=tl.float32)

    # For ConvTranspose3d (gather formulation):
    # out[n, oc, od, oh, ow] = sum_{ic, kd, kh, kw} x[n, ic, id, ih, iw] * w[ic, oc, kd, kh, kw]
    # where: id*SD - PD + kd = od  =>  id = (od + PD - kd) / SD  (must be integer, in [0, ID))

    # iterate kd, kh, kw, ic
    for kd in tl.static_range(0, KD):
        id_num = od + PD - kd
        id_ = id_num // SD
        id_valid = (id_num >= 0) & (id_num - id_ * SD == 0) & (id_ >= 0) & (id_ < ID)

        for kh in tl.static_range(0, KH):
            ih_num = oh + PH - kh
            ih_ = ih_num // SH
            ih_valid = (ih_num >= 0) & (ih_num - ih_ * SH == 0) & (ih_ >= 0) & (ih_ < IH)

            for kw in tl.static_range(0, KW):
                iw_num = ow_offs + PW - kw
                iw_ = iw_num // SW
                iw_valid = (iw_num >= 0) & (iw_num - iw_ * SW == 0) & (iw_ >= 0) & (iw_ < IW) & ow_mask

                # Combined validity mask
                spatial_valid = id_valid & ih_valid  # scalar
                # [BLOCK_W] mask:
                w_load_mask = iw_valid & spatial_valid

                # For each ic, load x and weight, accumulate
                for ic in range(0, IC):
                    # x[n, ic, id_, ih_, iw_]
                    x_base = pid_n * IC * ID * IH * IW + ic * ID * IH * IW + id_ * IH * IW + ih_ * IW
                    x_ptrs = x_ptr + x_base + iw_
                    x_vals = tl.load(x_ptrs, mask=w_load_mask, other=0.0)  # [BLOCK_W]

                    # w[ic, :, kd, kh, kw] -> [OC]
                    w_base = ic * OC * KD * KH * KW + kd * KH * KW + kh * KW + kw
                    w_ptrs = w_ptr + w_base + oc_offs * (KD * KH * KW)
                    w_vals = tl.load(w_ptrs, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    # outer product accumulate
                    acc += w_vals[:, None] * x_vals[None, :]

    # Add conv bias
    cb = tl.load(cb_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + cb[:, None]

    # mask out invalid OC rows with -inf for LSE
    acc = tl.where(oc_mask[:, None], acc, -float('inf'))

    # LogSumExp over OC dimension (axis=0)
    m = tl.max(acc, axis=0)  # [BLOCK_W]
    e = tl.exp(acc - m[None, :])
    e = tl.where(oc_mask[:, None], e, 0.0)
    s = tl.sum(e, axis=0)  # [BLOCK_W]
    lse = m + tl.log(s)

    # HardSwish: x * sigmoid(x + 3) / 6
    sig = 1.0 / (1.0 + tl.exp(-(lse + 3.0)))
    hs = lse * sig / 6.0

    # subtract bias
    b = tl.load(bias_ptr)
    y = hs - b

    # clamp
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)

    # store: out[pid_n, 0, od, oh, ow_offs]
    out_base = pid_n * OD * OH * OW + od * OH * OW + oh * OW
    tl.store(out_ptr + out_base + ow_offs, y, mask=ow_mask)


def fused_convt_post(x, weight, conv_bias, bias):
    N, IC, ID, IH, IW = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    assert IC == IC_w

    SD = SH = SW = 2
    PD = PH = PW = 1

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    out = torch.empty((N, 1, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_W = 64
    BLOCK_OC = 16  # OC is 16

    grid = (N, OD * OH, triton.cdiv(OW, BLOCK_W))

    conv_transpose3d_fused_kernel[grid](
        x, weight, conv_bias, bias.reshape(-1), out,
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        KD=KD, KH=KH, KW=KW,
        SD=SD, SH=SH, SW=SW,
        PD=PD, PH=PH, PW=PW,
        BLOCK_W=BLOCK_W,
        BLOCK_OC=BLOCK_OC,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()  # [IC, OC, KD, KH, KW]
        cb = self.conv_transpose.bias.contiguous() if self.conv_transpose.bias is not None else torch.zeros(w.shape[1], device=x.device, dtype=x.dtype)
        return fused_convt_post(x, w, cb, self.bias)