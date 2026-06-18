import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
    ],
    key=['IC', 'OC', 'OD', 'OH', 'OW'],
)
@triton.jit
def conv_transpose3d_kernel(
    x_ptr,        # [N, IC, ID, IH, IW]
    w_ptr,        # [IC, OC, KD, KH, KW]
    b_ptr,        # [OC]  conv bias
    eb_ptr,       # [OC]  extra bias for epilogue
    out_ptr,      # [N, OC, OD, OH, OW]
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    OHW = OH * OW
    ODHW = OD * OHW

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < ODHW

    od = sp_offs // OHW
    rem = sp_offs - od * OHW
    oh = rem // OW
    ow = rem - oh * OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    k_offs = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    x_base_n = pid_n * (IC * ID * IH * IW)
    IDHW = ID * IH * IW
    IHW = IH * IW
    KDHW = KD * KH * KW

    for kd in tl.static_range(0, KD):
        id_num = od + PD - kd
        id_q = id_num // SD
        id_ok = ((id_num - id_q * SD) == 0) & (id_q >= 0) & (id_q < ID)

        for kh in tl.static_range(0, KH):
            ih_num = oh + PH - kh
            ih_q = ih_num // SH
            ih_ok = ((ih_num - ih_q * SH) == 0) & (ih_q >= 0) & (ih_q < IH)

            for kw in tl.static_range(0, KW):
                iw_num = ow + PW - kw
                iw_q = iw_num // SW
                iw_ok = ((iw_num - iw_q * SW) == 0) & (iw_q >= 0) & (iw_q < IW)

                spatial_ok = id_ok & ih_ok & iw_ok & sp_mask  # [BLOCK_SP]

                in_spatial = id_q * IHW + ih_q * IW + iw_q  # [BLOCK_SP]
                w_kpos = kd * (KH * KW) + kh * KW + kw

                # Tile over IC in chunks of BLOCK_K
                for ic_start in range(0, IC, BLOCK_K):
                    ic_idx = ic_start + k_offs  # [BLOCK_K]
                    ic_mask = ic_idx < IC

                    # x[n, ic_idx, id_q, ih_q, iw_q] -> [BLOCK_SP, BLOCK_K]
                    x_off = (x_base_n
                             + ic_idx[None, :] * IDHW
                             + in_spatial[:, None])
                    x_m = spatial_ok[:, None] & ic_mask[None, :]
                    x_tile = tl.load(x_ptr + x_off, mask=x_m, other=0.0)

                    # w[ic_idx, oc_offs, kd, kh, kw] -> [BLOCK_K, BLOCK_OC]
                    w_off = (ic_idx[:, None] * (OC * KDHW)
                             + oc_offs[None, :] * KDHW
                             + w_kpos)
                    w_m = ic_mask[:, None] & oc_mask[None, :]
                    w_tile = tl.load(w_ptr + w_off, mask=w_m, other=0.0)

                    acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    # Add conv bias
    bias_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    y = acc + bias_val[None, :]

    # Fused epilogue: out = (2*y + extra_bias)*y + y
    eb_val = tl.load(eb_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    out_val = (2.0 * y + eb_val[None, :]) * y + y

    out_offset = (
        pid_n * (OC * ODHW)
        + oc_offs[None, :] * ODHW
        + sp_offs[:, None]
    )
    out_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_offset, out_val, mask=out_mask)


@triton.jit
def fused_epilogue_kernel(
    x_ptr, bias_ptr, out_ptr,
    C, S,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    c_idx = (offsets // S) % C
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    out = (2.0 * x + b) * x + x
    tl.store(out_ptr + offsets, out, mask=mask)


def conv_transpose3d_triton(x, weight, conv_bias, extra_bias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    SD, SH, SW = stride, stride, stride
    PD, PH, PW = padding, padding, padding

    OD = (ID - 1) * SD - 2 * PD + KD + (1 if SD > 1 else 0)
    OH = (IH - 1) * SH - 2 * PH + KH + (1 if SH > 1 else 0)
    OW = (IW - 1) * SW - 2 * PW + KW + (1 if SW > 1 else 0)

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_K = 32

    grid = lambda META: (
        N,
        triton.cdiv(OC, META['BLOCK_OC']),
        triton.cdiv(OD * OH * OW, META['BLOCK_SP']),
    )

    conv_transpose3d_kernel[grid](
        x, weight, conv_bias, extra_bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_K=BLOCK_K,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous()
        weight = self.conv_transpose.weight  # [IC, OC, KD, KH, KW]
        conv_bias = self.conv_transpose.bias
        if conv_bias is None:
            conv_bias = torch.zeros(weight.shape[1], device=x.device, dtype=x.dtype)

        extra_bias = self.bias.contiguous().view(-1)

        out = conv_transpose3d_triton(
            x, weight.contiguous(), conv_bias.contiguous(),
            extra_bias, self.stride, self.padding,
        )
        return out