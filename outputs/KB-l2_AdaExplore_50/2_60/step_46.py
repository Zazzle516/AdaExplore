import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_scatter_kernel(
    x_ptr, w_ptr, y_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    IC_CONST: tl.constexpr,
    OC_CONST: tl.constexpr,
    BLOCK_IW: tl.constexpr,
):
    # grid: (N * ID * IH, ceil(IW/BLOCK_IW))
    pid0 = tl.program_id(0)
    pid_w = tl.program_id(1)

    n = pid0 // (ID * IH)
    rem = pid0 % (ID * IH)
    id_ = rem // IH
    ih_ = rem % IH

    iw_offs = pid_w * BLOCK_IW + tl.arange(0, BLOCK_IW)  # [BLOCK_IW]
    iw_mask = iw_offs < IW

    # Load x[n, :, id_, ih_, iw_offs] -> [IC, BLOCK_IW]
    ic_range = tl.arange(0, IC_CONST)  # [IC]
    oc_range = tl.arange(0, OC_CONST)  # [OC]

    # x base: n*IC*ID*IH*IW + ic*ID*IH*IW + id_*IH*IW + ih_*IW + iw
    x_base = n * IC * ID * IH * IW + id_ * IH * IW + ih_ * IW
    x_ptrs = x_ptr + x_base + ic_range[:, None] * (ID * IH * IW) + iw_offs[None, :]
    x_vals = tl.load(x_ptrs, mask=iw_mask[None, :], other=0.0)  # [IC, BLOCK_IW]

    # output spatial bases
    od_base = id_ * SD - PD  # then + kd
    oh_base = ih_ * SH - PH
    ow_base = iw_offs * SW - PW  # [BLOCK_IW]

    # For each (kd, kh, kw) compute weight slice and accumulate output position
    for kd in tl.static_range(0, KD):
        od = od_base + kd
        d_ok = (od >= 0) & (od < OD)
        for kh in tl.static_range(0, KH):
            oh = oh_base + kh
            h_ok = (oh >= 0) & (oh < OH)
            for kw in tl.static_range(0, KW):
                ow = ow_base + kw  # [BLOCK_IW]
                w_ok = (ow >= 0) & (ow < OW) & iw_mask
                spatial_ok = d_ok & h_ok & w_ok  # [BLOCK_IW]

                # weight[ic, oc, kd, kh, kw]; shape (IC, OC, KD, KH, KW)
                w_ptrs = (w_ptr
                          + ic_range[:, None] * (OC * KD * KH * KW)
                          + oc_range[None, :] * (KD * KH * KW)
                          + kd * KH * KW + kh * KW + kw)
                w_vals = tl.load(w_ptrs)  # [IC, OC]

                # outer product over IC: out[oc, iw] = sum_ic x[ic, iw] * w[ic, oc]
                # compute as matmul: w_vals.T @ x_vals -> [OC, BLOCK_IW]
                # we can use tl.dot
                out_tile = tl.dot(tl.trans(w_vals), x_vals)  # [OC, BLOCK_IW]

                # scatter-add to y[n, oc, od, oh, ow]
                y_ptrs = (y_ptr
                          + n * (OC * OD * OH * OW)
                          + oc_range[:, None] * (OD * OH * OW)
                          + od * (OH * OW) + oh * OW + ow[None, :])
                tl.atomic_add(y_ptrs, out_tile, mask=spatial_ok[None, :])


@triton.jit
def swish_groupnorm_hardswish_kernel(
    x_ptr, y_ptr, bias_ptr,
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
        c_idx = g * C_PER_G_CONST + c_inner
        b = tl.load(bias_ptr + c_idx).to(tl.float32)
        c_off = base + c_inner * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            x = tl.load(x_ptr + c_off + offs, mask=mask, other=0.0).to(tl.float32)
            x = x + b
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
        b = tl.load(bias_ptr + c_idx).to(tl.float32)
        c_off = base + c_inner * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            x = tl.load(x_ptr + c_off + offs, mask=mask, other=0.0).to(tl.float32)
            x = x + b
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

        # Allocate output with zeros for atomic_add
        y_conv = torch.zeros((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

        # Pick BLOCK_IW (power of 2, >=16 for tl.dot)
        BLOCK_IW = 32
        if IW < 32:
            BLOCK_IW = max(triton.next_power_of_2(IW), 16)

        grid = (N * ID * IH, triton.cdiv(IW, BLOCK_IW))
        conv_transpose3d_scatter_kernel[grid](
            x, weight, y_conv,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            IC_CONST=IC,
            OC_CONST=OC,
            BLOCK_IW=BLOCK_IW,
            num_warps=4,
            num_stages=2,
        )

        # bias: passed separately into norm kernel to fuse the addition
        if self.conv_transpose.bias is not None:
            bias_t = self.conv_transpose.bias.contiguous()
        else:
            bias_t = torch.zeros(OC, device=x.device, dtype=torch.float32)

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
            y_conv, out, bias_t,
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