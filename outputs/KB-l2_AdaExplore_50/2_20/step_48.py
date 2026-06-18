import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_gather_kernel(
    x_ptr, w_ptr, cb_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    OHW = OH * OW
    ODHW = OD * OHW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < ODHW

    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # Loop over kernel positions and input channels
    for kd in tl.static_range(0, KD):
        id_pos = od + PD - kd
        id_in = id_pos // SD
        id_valid = (id_pos % SD == 0) & (id_in >= 0) & (id_in < ID)
        for kh in tl.static_range(0, KH):
            ih_pos = oh + PH - kh
            ih_in = ih_pos // SH
            ih_valid = (ih_pos % SH == 0) & (ih_in >= 0) & (ih_in < IH)
            for kw in tl.static_range(0, KW):
                iw_pos = ow + PW - kw
                iw_in = iw_pos // SW
                iw_valid = (iw_pos % SW == 0) & (iw_in >= 0) & (iw_in < IW)
                sp_valid = id_valid & ih_valid & iw_valid & sp_mask
                # input spatial offset within (D,H,W)
                in_sp_off = id_in * (IH * IW) + ih_in * IW + iw_in
                # weight offset starts: weight shape [IC, OC, KD, KH, KW]
                # We'll loop over IC
                k_off = kd * (KH * KW) + kh * KW + kw  # within (KD,KH,KW)
                for ic in range(0, IC):
                    # input ptr: x[pid_n, ic, id_in, ih_in, iw_in]
                    in_ptr = x_ptr + pid_n * (IC * ID * IH * IW) + ic * (ID * IH * IW) + in_sp_off
                    x_val = tl.load(in_ptr, mask=sp_valid, other=0.0)
                    # weight ptr: w[ic, oc_offs, kd, kh, kw]
                    w_ptrs = w_ptr + ic * (OC * KD * KH * KW) + oc_offs * (KD * KH * KW) + k_off
                    w_val = tl.load(w_ptrs, mask=oc_mask, other=0.0)
                    acc += w_val[:, None] * x_val[None, :]

    # Add conv bias
    cb = tl.load(cb_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += cb[:, None]

    # Add learned bias (shape [OC]) - broadcast over spatial
    b = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    # Epilogue: out = (2x + b) * x + x  where x = acc, b is learned bias
    bcast = b[:, None]
    out_val = (2.0 * acc + bcast) * acc + acc

    # Store
    out_offs = pid_n * (OC * ODHW) + oc_offs[:, None] * ODHW + sp_offs[None, :]
    mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_offs, out_val, mask=mask)


def conv_transpose3d_fused(x, weight, conv_bias, bias,
                            stride, padding, output_padding):
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    SD = SH = SW = stride
    PD = PH = PW = padding
    OD = (ID - 1) * SD - 2 * PD + KD + output_padding
    OH = (IH - 1) * SH - 2 * PH + KH + output_padding
    OW = (IW - 1) * SW - 2 * PW + KW + output_padding

    x = x.contiguous()
    weight = weight.contiguous()
    conv_bias = conv_bias.contiguous()
    bias_flat = bias.contiguous().view(-1)

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 32
    BLOCK_SP = 128

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OD * OH * OW, BLOCK_SP))

    conv_transpose3d_gather_kernel[grid](
        x, weight, conv_bias, bias_flat, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        return conv_transpose3d_fused(
            x, self.conv_transpose.weight, self.conv_transpose.bias, self.bias,
            self.stride, self.padding, self.output_padding,
        )