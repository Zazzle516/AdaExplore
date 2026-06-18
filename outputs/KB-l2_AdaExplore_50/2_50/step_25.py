import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OW': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_OW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_OW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 128}, num_warps=8, num_stages=3),
    ],
    key=['N', 'OC', 'OD', 'OH', 'OW', 'IC', 'KD', 'KH', 'KW'],
)
@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    pid_noc = tl.program_id(0)
    pid_dh = tl.program_id(1)
    pid_w = tl.program_id(2)

    n = pid_noc // OC
    oc = pid_noc % OC
    od = pid_dh // OH
    oh = pid_dh % OH

    ow_offs = pid_w * BLOCK_OW + tl.arange(0, BLOCK_OW)
    ow_mask = ow_offs < OW

    acc = tl.zeros((BLOCK_OW,), dtype=tl.float32)

    for kd in tl.static_range(KD):
        pd_num = od + PD - kd
        id_pos = pd_num // SD
        id_valid = (pd_num >= 0) & (pd_num % SD == 0) & (id_pos >= 0) & (id_pos < ID)
        for kh in tl.static_range(KH):
            ph_num = oh + PH - kh
            ih_pos = ph_num // SH
            ih_valid = (ph_num >= 0) & (ph_num % SH == 0) & (ih_pos >= 0) & (ih_pos < IH)
            dh_valid = id_valid & ih_valid
            for kw in tl.static_range(KW):
                pw_num = ow_offs + PW - kw
                iw_pos = pw_num // SW
                iw_valid = (pw_num >= 0) & (pw_num % SW == 0) & (iw_pos >= 0) & (iw_pos < IW)
                valid = dh_valid & iw_valid & ow_mask
                base_x = (n * IC) * ID * IH * IW + id_pos * IH * IW + ih_pos * IW + iw_pos
                base_w = (oc * KD + kd) * KH * KW + kh * KW + kw
                for ic in tl.static_range(IC):
                    x_off = base_x + ic * ID * IH * IW
                    x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                    w_off = base_w + ic * OC * KD * KH * KW
                    w_val = tl.load(w_ptr + w_off)
                    acc += x_val * w_val

    b_val = tl.load(b_ptr + oc)
    acc += b_val

    out_off = ((n * OC + oc) * OD + od) * OH * OW + oh * OW + ow_offs
    tl.store(out_ptr + out_off, acc, mask=ow_mask)


@triton.jit
def fused_pool_bias_scale_kernel(
    in_ptr, bias_ptr, out_ptr,
    scale_combined,
    N, C, D, H, W,
    PD, PH, PW,
    BLOCK_HW: tl.constexpr,
    PH_C: tl.constexpr,
    PW_C: tl.constexpr,
):
    # One program per (N*C, PD). It processes the entire (PH, PW) plane.
    pid_nc = tl.program_id(0)
    pd = tl.program_id(1)

    n = pid_nc // C
    c = pid_nc % C

    offs = tl.arange(0, BLOCK_HW)
    ph_idx = offs // PW_C
    pw_idx = offs % PW_C
    mask = offs < (PH_C * PW_C)

    # 2x2x2 avg pool input base
    in_d = 2 * pd
    in_h = 2 * ph_idx
    in_w = 2 * pw_idx
    base = ((n * C + c) * D + in_d) * H * W + in_h * W + in_w

    acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)
    for dd in tl.static_range(2):
        for hh in tl.static_range(2):
            for ww in tl.static_range(2):
                off = base + dd * H * W + hh * W + ww
                v = tl.load(in_ptr + off, mask=mask, other=0.0)
                acc += v
    acc = acc * (scale_combined * 0.125)

    b = tl.load(bias_ptr + c)
    acc = acc + b * scale_combined

    out_base = ((n * C + c) * PD + pd) * PH_C * PW_C
    tl.store(out_ptr + out_base + offs, acc, mask=mask)


def conv_transpose3d_triton(x, weight, bias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    assert IC == IC_w
    SD, SH, SW = stride
    PD, PH, PW = padding

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

    grid = lambda META: (N * OC, OD * OH, (OW + META['BLOCK_OW'] - 1) // META['BLOCK_OW'])
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
    )
    return out


def fused_pool_bias_scale(x, bias, scale_combined):
    N, C, D, H, W = x.shape
    PD, PH, PW = D // 2, H // 2, W // 2
    out = torch.empty((N, C, PD, PH, PW), device=x.device, dtype=torch.float32)
    # one program per (N*C, PD); process entire (PH, PW) plane in BLOCK_HW threads
    block_hw = 1
    while block_hw < PH * PW:
        block_hw *= 2
    grid = (N * C, PD)
    nw = 4 if block_hw >= 128 else 2
    fused_pool_bias_scale_kernel[grid](
        x, bias, out,
        float(scale_combined),
        N, C, D, H, W,
        PD, PH, PW,
        BLOCK_HW=block_hw,
        PH_C=PH,
        PW_C=PW,
        num_warps=nw,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous()
        # Fold scale1 into bias of conv (we want conv_out * scale1 + (avgpool then) ...)
        # Actually we apply scale1 after conv. Fold scale1 into weight & bias for fewer ops.
        scaled_weight = weight * self.scale1
        scaled_bias = self.conv_transpose.bias * self.scale1

        ks = self.kernel_size
        if isinstance(ks, int):
            ks = (ks, ks, ks)
        st = self.stride
        if isinstance(st, int):
            st = (st, st, st)
        pd = self.padding
        if isinstance(pd, int):
            pd = (pd, pd, pd)

        conv_out = conv_transpose3d_triton(x, scaled_weight, scaled_bias, st, pd)

        # avg_pool + bias + scale2 fused
        bias_flat = self.bias.view(-1).contiguous()
        scale2_val = self.scale2.item()
        out = fused_pool_bias_scale(conv_out, bias_flat, scale2_val)
        return out