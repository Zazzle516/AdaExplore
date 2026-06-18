import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    min_value, inv_divisor,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    OHW = OH * OW
    ODHW = OD * OHW

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < ODHW

    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # x: [N, IC, ID, IH, IW], stride: (IC*ID*IH*IW, ID*IH*IW, IH*IW, IW, 1)
    # w: [IC, OC, KD, KH, KW], stride: (OC*KD*KH*KW, KD*KH*KW, KH*KW, KW, 1)
    x_n_base = pid_n * IC * ID * IH * IW

    for kd in tl.static_range(KD):
        id_pos = od + PAD - kd
        id_valid = (id_pos % STRIDE) == 0
        id_idx = id_pos // STRIDE
        id_in_range = (id_idx >= 0) & (id_idx < ID)
        d_ok = id_valid & id_in_range  # [BLOCK_SP]

        for kh in tl.static_range(KH):
            ih_pos = oh + PAD - kh
            ih_valid = (ih_pos % STRIDE) == 0
            ih_idx = ih_pos // STRIDE
            ih_in_range = (ih_idx >= 0) & (ih_idx < IH)
            h_ok = ih_valid & ih_in_range

            for kw in tl.static_range(KW):
                iw_pos = ow + PAD - kw
                iw_valid = (iw_pos % STRIDE) == 0
                iw_idx = iw_pos // STRIDE
                iw_in_range = (iw_idx >= 0) & (iw_idx < IW)
                w_ok = iw_valid & iw_in_range

                spatial_ok = d_ok & h_ok & w_ok & sp_mask  # [BLOCK_SP]
                x_spatial_offset = id_idx * (IH * IW) + ih_idx * IW + iw_idx  # [BLOCK_SP]

                for ic_start in range(0, IC, BLOCK_IC):
                    ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                    ic_mask = ic_offs < IC

                    # Load x[N, ic, id, ih, iw] -> [BLOCK_SP, BLOCK_IC]
                    x_offsets = (x_n_base
                                 + ic_offs[None, :] * (ID * IH * IW)
                                 + x_spatial_offset[:, None])
                    x_load_mask = spatial_ok[:, None] & ic_mask[None, :]
                    x_vals = tl.load(x_ptr + x_offsets, mask=x_load_mask, other=0.0)

                    # Load w[ic, oc, kd, kh, kw] -> [BLOCK_IC, BLOCK_OC]
                    w_offsets = (ic_offs[:, None] * (OC * KD * KH * KW)
                                 + oc_offs[None, :] * (KD * KH * KW)
                                 + kd * (KH * KW) + kh * KW + kw)
                    w_load_mask = ic_mask[:, None] & oc_mask[None, :]
                    w_vals = tl.load(w_ptr + w_offsets, mask=w_load_mask, other=0.0)

                    acc += tl.dot(x_vals, w_vals)

    # Add bias and apply clamp+div
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[None, :]
    acc = tl.where(acc < min_value, min_value, acc)
    acc = acc * inv_divisor

    out_offsets = (pid_n * OC * ODHW
                   + oc_offs[None, :] * ODHW
                   + sp_offs[:, None])
    out_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=out_mask)


def conv_transpose3d_triton(x, weight, bias, stride, padding, min_value, divisor):
    N, IC, ID, IH, IW = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    assert IC == IC_w

    OD = (ID - 1) * stride - 2 * padding + KD
    OH = (IH - 1) * stride - 2 * padding + KH
    OW = (IW - 1) * stride - 2 * padding + KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    BLOCK_OC = 32
    BLOCK_SP = 64
    BLOCK_IC = 32

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OD * OH * OW, BLOCK_SP))

    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        stride, padding,
        float(min_value), 1.0 / float(divisor),
        BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP, BLOCK_IC=BLOCK_IC,
        num_warps=4, num_stages=2,
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
        return conv_transpose3d_triton(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.stride,
            self.padding,
            self.min_value,
            self.divisor,
        )