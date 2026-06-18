import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_gather_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    min_value, inv_divisor,
    BLOCK_OC: tl.constexpr, BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    OHW = OH * OW
    SP = OD * OH * OW

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < SP

    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # input strides
    x_n_stride = IC * ID * IH * IW
    x_c_stride = ID * IH * IW
    x_d_stride = IH * IW
    x_h_stride = IW

    # weight layout: (IC, OC, KD, KH, KW)
    w_ic_stride = OC * KD * KH * KW
    w_oc_stride = KD * KH * KW

    for kd in tl.static_range(0, KD):
        id_num = od + PD - kd
        id_ = id_num // SD
        id_valid = ((id_num - id_ * SD) == 0) & (id_ >= 0) & (id_ < ID)
        for kh in tl.static_range(0, KH):
            ih_num = oh + PH - kh
            ih_ = ih_num // SH
            ih_valid = ((ih_num - ih_ * SH) == 0) & (ih_ >= 0) & (ih_ < IH)
            for kw in tl.static_range(0, KW):
                iw_num = ow + PW - kw
                iw_ = iw_num // SW
                iw_valid = ((iw_num - iw_ * SW) == 0) & (iw_ >= 0) & (iw_ < IW)

                spatial_valid = id_valid & ih_valid & iw_valid & sp_mask

                in_base = (pid_n * x_n_stride
                           + id_ * x_d_stride
                           + ih_ * x_h_stride
                           + iw_)  # shape [BLOCK_SP]

                w_base = kd * (KH * KW) + kh * KW + kw  # scalar

                for ic in range(0, IC):
                    x_ptrs = x_ptr + in_base + ic * x_c_stride
                    x_vals = tl.load(x_ptrs, mask=spatial_valid, other=0.0)  # [BLOCK_SP]

                    w_ptrs = w_ptr + ic * w_ic_stride + oc_offs * w_oc_stride + w_base
                    w_vals = tl.load(w_ptrs, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    acc += w_vals[:, None] * x_vals[None, :]

    # bias
    b = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + b[:, None]

    # clamp + div
    acc = tl.maximum(acc, min_value)
    acc = acc * inv_divisor

    # store: output shape (N, OC, OD, OH, OW)
    out_base = pid_n * (OC * SP) + oc_offs[:, None] * SP + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_base, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = float(min_value)
        self.divisor = float(divisor)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        SD = SH = SW = self.stride
        PD = PH = PW = self.padding
        OC = self.out_channels

        OD = (ID - 1) * SD - 2 * PD + KD
        OH = (IH - 1) * SH - 2 * PH + KH
        OW = (IW - 1) * SW - 2 * PW + KW

        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, KD, KH, KW)
        bias = self.conv_transpose.bias.contiguous()

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_SP = 128

        SP = OD * OH * OW
        grid = (N, (OC + BLOCK_OC - 1) // BLOCK_OC, (SP + BLOCK_SP - 1) // BLOCK_SP)

        conv_transpose3d_gather_kernel[grid](
            x, weight, bias, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            self.min_value, 1.0 / self.divisor,
            BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
            num_warps=4, num_stages=2,
        )
        return out