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
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    BLOCK_OW: tl.constexpr,
):
    # grid: (N * OC, OD * OH, ceil(OW / BLOCK_OW))
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

    # output position p = stride*i - pad + k  => i = (p + pad - k) / stride
    # For each k in kernel, find valid i.
    for kd in range(KD):
        pd_num = od + PD - kd
        id_pos = pd_num // SD
        id_valid = (pd_num >= 0) & (pd_num % SD == 0) & (id_pos >= 0) & (id_pos < ID)
        for kh in range(KH):
            ph_num = oh + PH - kh
            ih_pos = ph_num // SH
            ih_valid = (ph_num >= 0) & (ph_num % SH == 0) & (ih_pos >= 0) & (ih_pos < IH)
            for kw in range(KW):
                pw_num = ow_offs + PW - kw
                iw_pos = pw_num // SW
                iw_valid = (pw_num >= 0) & (pw_num % SW == 0) & (iw_pos >= 0) & (iw_pos < IW)
                valid = id_valid & ih_valid & iw_valid & ow_mask
                # Loop over input channels
                for ic in range(IC):
                    # x[n, ic, id_pos, ih_pos, iw_pos]
                    x_off = ((n * IC + ic) * ID + id_pos) * IH * IW + ih_pos * IW + iw_pos
                    x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                    # weight is [IC, OC, KD, KH, KW]
                    w_off = ((ic * OC + oc) * KD + kd) * KH * KW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off)
                    acc += x_val * w_val

    # add bias
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
    BLOCK: tl.constexpr,
):
    # one program per (N*C, PD*PH) block, vectorize over PW
    pid_nc = tl.program_id(0)
    pid_dh = tl.program_id(1)

    n = pid_nc // C
    c = pid_nc % C
    pd = pid_dh // PH
    ph = pid_dh % PH

    pw_offs = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    pw_mask = pw_offs < PW

    # 2x2x2 average pool from input position (2*pd, 2*ph, 2*pw)
    base = ((n * C + c) * D + 2 * pd) * H * W + (2 * ph) * W
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for dd in range(2):
        for hh in range(2):
            for ww in range(2):
                off = base + dd * H * W + hh * W + ww + 2 * pw_offs
                v = tl.load(in_ptr + off, mask=pw_mask, other=0.0)
                acc += v
    acc = acc / 8.0

    b = tl.load(bias_ptr + c)
    acc = (acc + b) * scale_combined

    out_off = ((n * C + c) * PD + pd) * PH * PW + ph * PW + pw_offs
    tl.store(out_ptr + out_off, acc, mask=pw_mask)


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

    BLOCK_OW = 32
    grid = (N * OC, OD * OH, (OW + BLOCK_OW - 1) // BLOCK_OW)
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_OW=BLOCK_OW,
        num_warps=4,
    )
    return out


def fused_pool_bias_scale(x, bias, scale_combined):
    N, C, D, H, W = x.shape
    PD, PH, PW = D // 2, H // 2, W // 2
    out = torch.empty((N, C, PD, PH, PW), device=x.device, dtype=torch.float32)
    BLOCK = 16
    grid = (N * C, PD * PH, (PW + BLOCK - 1) // BLOCK)
    fused_pool_bias_scale_kernel[grid](
        x, bias, out,
        float(scale_combined),
        N, C, D, H, W,
        PD, PH, PW,
        BLOCK=BLOCK,
        num_warps=2,
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