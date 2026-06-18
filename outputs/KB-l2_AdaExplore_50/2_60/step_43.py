import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    BLOCK_W: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # grid: (N * OD * OH, ceil(OW/BLOCK_W), ceil(OC/BLOCK_OC))
    pid0 = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_oc = tl.program_id(2)

    n = pid0 // (OD * OH)
    rem = pid0 % (OD * OH)
    od = rem // OH
    oh = rem % OH

    ow_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    ow_mask = ow_offs < OW
    oc_mask = oc_offs < OC

    # accumulator [BLOCK_OC, BLOCK_W]
    acc = tl.zeros([BLOCK_OC, BLOCK_W], dtype=tl.float32)

    # for each output position, gather contributions
    # iod_num = od + PD - kd ; must be divisible by SD, then id = iod_num/SD in [0, ID)
    for kd in range(KD):
        iod_num = od + PD - kd
        id_q = iod_num // SD
        id_r = iod_num - id_q * SD
        d_ok = (id_r == 0) & (id_q >= 0) & (id_q < ID)
        for kh in range(KH):
            ioh_num = oh + PH - kh
            ih_q = ioh_num // SH
            ih_r = ioh_num - ih_q * SH
            h_ok = (ih_r == 0) & (ih_q >= 0) & (ih_q < IH)
            for kw in range(KW):
                iow_num = ow_offs + PW - kw
                iw_q = iow_num // SW
                iw_r = iow_num - iw_q * SW
                w_ok = (iw_r == 0) & (iw_q >= 0) & (iw_q < IW) & ow_mask

                spatial_ok = d_ok & h_ok & w_ok  # [BLOCK_W]

                # input index for n, ic varying, id_q, ih_q, iw_q
                # x shape: (N, IC, ID, IH, IW)
                # weight shape: (IC, OC, KD, KH, KW)
                base_x = n * IC * ID * IH * IW + id_q * IH * IW + ih_q * IW + iw_q  # [BLOCK_W]

                for ic in range(IC):
                    # load x[n, ic, id_q, ih_q, iw_q] for each ow
                    x_ptrs = x_ptr + ic * ID * IH * IW + base_x  # [BLOCK_W]
                    x_vals = tl.load(x_ptrs, mask=spatial_ok, other=0.0)  # [BLOCK_W]

                    # load w[ic, oc_offs, kd, kh, kw]
                    w_ptrs = w_ptr + ic * OC * KD * KH * KW + oc_offs * KD * KH * KW + kd * KH * KW + kh * KW + kw
                    w_vals = tl.load(w_ptrs, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    acc += w_vals[:, None] * x_vals[None, :]

    # add bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += bias[:, None]

    # store to y[n, oc_offs, od, oh, ow_offs]
    y_base = n * OC * OD * OH * OW + oc_offs[:, None] * OD * OH * OW + od * OH * OW + oh * OW + ow_offs[None, :]
    mask_out = oc_mask[:, None] & ow_mask[None, :]
    tl.store(y_ptr + y_base, acc, mask=mask_out)


@triton.jit
def swish_groupnorm_hardswish_kernel(
    x_ptr, y_ptr,
    gamma_ptr, beta_ptr,
    C, S,
    G,
    eps,
    inv_group_size,
    BLOCK_S: tl.constexpr,
    C_PER_G_CONST: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    base = n * C * S + g * C_PER_G_CONST * S

    sum_val = tl.zeros([], dtype=tl.float32)
    sum_sq = tl.zeros([], dtype=tl.float32)

    for c_inner in range(0, C_PER_G_CONST):
        c_off = base + c_inner * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            x = tl.load(x_ptr + c_off + offs, mask=mask, other=0.0).to(tl.float32)
            sw = x * tl.sigmoid(x)
            sw = tl.where(mask, sw, 0.0)
            sum_val += tl.sum(sw, axis=0)
            sum_sq += tl.sum(sw * sw, axis=0)

    mean = sum_val * inv_group_size
    var = sum_sq * inv_group_size - mean * mean
    rstd = tl.rsqrt(var + eps)

    for c_inner in range(0, C_PER_G_CONST):
        c_idx = g * C_PER_G_CONST + c_inner
        gamma = tl.load(gamma_ptr + c_idx).to(tl.float32)
        beta = tl.load(beta_ptr + c_idx).to(tl.float32)
        c_off = base + c_inner * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            x = tl.load(x_ptr + c_off + offs, mask=mask, other=0.0).to(tl.float32)
            sw = x * tl.sigmoid(x)
            normed = (sw - mean) * rstd
            y = normed * gamma + beta
            t = y + 3.0
            t = tl.minimum(tl.maximum(t, 0.0), 6.0)
            out = y * t * (1.0 / 6.0)
            tl.store(y_ptr + c_off + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.groups = groups
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.eps = eps

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        SD = SH = SW = self.stride
        PD = PH = PW = self.padding

        OD = (ID - 1) * SD - 2 * PD + KD
        OH = (IH - 1) * SH - 2 * PH + KH
        OW = (IW - 1) * SW - 2 * PW + KW

        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, KD, KH, KW)
        bias = self.conv_transpose.bias.contiguous() if self.conv_transpose.bias is not None else torch.zeros(OC, device=x.device, dtype=x.dtype)

        y_conv = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_W = 64 if OW >= 64 else triton.next_power_of_2(OW)
        BLOCK_OC = 16 if OC >= 16 else triton.next_power_of_2(OC)

        grid = (N * OD * OH, triton.cdiv(OW, BLOCK_W), triton.cdiv(OC, BLOCK_OC))
        conv_transpose3d_kernel[grid](
            x, weight, bias, y_conv,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            BLOCK_W=BLOCK_W,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
            num_stages=2,
        )

        C = OC
        S = OD * OH * OW
        out = torch.empty_like(y_conv)
        G = self.groups
        C_PER_G = C // G

        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()

        if S >= 2048:
            BLOCK_S = 2048
            num_warps = 8
        elif S >= 1024:
            BLOCK_S = 1024
            num_warps = 8
        else:
            BLOCK_S = max(triton.next_power_of_2(S), 64)
            num_warps = 4

        inv_group_size = 1.0 / float(C_PER_G * S)

        grid2 = (N * G,)
        swish_groupnorm_hardswish_kernel[grid2](
            y_conv, out,
            gamma, beta,
            C, S,
            G,
            self.eps,
            inv_group_size,
            BLOCK_S=BLOCK_S,
            C_PER_G_CONST=C_PER_G,
            num_warps=num_warps,
            num_stages=2,
        )
        return out