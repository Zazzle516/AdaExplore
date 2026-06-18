import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_swish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # grid: (N * OC, OD * OH, ceil(OW / BLOCK_W))
    pid_nc = tl.program_id(0)
    pid_dh = tl.program_id(1)
    pid_w = tl.program_id(2)

    n = pid_nc // OC
    oc = pid_nc % OC
    od = pid_dh // OH
    oh = pid_dh % OH

    ow_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    ow_mask = ow_offs < OW

    # For ConvTranspose: output[n, oc, od, oh, ow] = sum over (ic, kd, kh, kw):
    #   input[n, ic, id, ih, iw] * weight[ic, oc, kd, kh, kw]
    # where: od = id*SD - PD + kd  =>  id = (od + PD - kd) / SD,  needs divisible
    #        oh = ih*SH - PH + kh  =>  ih = (oh + PH - kh) / SH
    #        ow = iw*SW - PW + kw  =>  iw = (ow + PW - kw) / SW

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    od_p_pd = od + PD
    oh_p_ph = oh + PH
    ow_p_pw = ow_offs + PW  # [BLOCK_W]

    for kd in tl.static_range(0, KD):
        id_num = od_p_pd - kd
        id_ = id_num // SD
        id_valid = (id_num >= 0) & ((id_num - id_ * SD) == 0) & (id_ < ID) & (id_ >= 0)
        for kh in tl.static_range(0, KH):
            ih_num = oh_p_ph - kh
            ih_ = ih_num // SH
            ih_valid = (ih_num >= 0) & ((ih_num - ih_ * SH) == 0) & (ih_ < IH) & (ih_ >= 0)
            dh_valid = id_valid & ih_valid
            for kw in tl.static_range(0, KW):
                iw_num = ow_p_pw - kw  # [BLOCK_W]
                iw_ = iw_num // SW
                iw_valid_base = (iw_num >= 0) & ((iw_num - iw_ * SW) == 0) & (iw_ < IW) & (iw_ >= 0)
                v_mask = ow_mask & dh_valid & iw_valid_base

                # Loop over IC
                for ic in range(0, IC):
                    # input ptr: [n, ic, id_, ih_, iw_]
                    in_off = ((n * IC + ic) * ID + id_) * IH * IW + ih_ * IW + iw_
                    x_val = tl.load(x_ptr + in_off, mask=v_mask, other=0.0)
                    # weight ptr: [ic, oc, kd, kh, kw]
                    w_off = ((ic * OC + oc) * KD + kd) * KH * KW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off)
                    acc += x_val * w_val

    # Add bias
    bias = tl.load(b_ptr + oc)
    acc = acc + bias

    # Swish: x * sigmoid(x)
    sig = tl.sigmoid(acc)
    out = acc * sig

    # Store
    out_off = (((n * OC + oc) * OD + od) * OH + oh) * OW + ow_offs
    tl.store(out_ptr + out_off, out, mask=ow_mask)


@triton.jit
def group_norm_hardswish_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C, S, G, C_per_G, eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    group_elems = C_per_G * S
    base = n * C * S + g * C_per_G * S

    sum_val = tl.zeros([1], dtype=tl.float32)
    sum_sq = tl.zeros([1], dtype=tl.float32)
    offs = tl.arange(0, BLOCK_SIZE)

    for start in range(0, group_elems, BLOCK_SIZE):
        idx = start + offs
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / group_elems
    var = sum_sq / group_elems - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for start in range(0, group_elems, BLOCK_SIZE):
        idx = start + offs
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)

        c_in_g = idx // S
        c_global = g * C_per_G + c_in_g
        w = tl.load(weight_ptr + c_global, mask=mask, other=0.0)
        b = tl.load(bias_ptr + c_global, mask=mask, other=0.0)

        y = (x - mean) * rstd * w + b
        t = y + 3.0
        t = tl.minimum(tl.maximum(t, 0.0), 6.0)
        out = y * t / 6.0
        tl.store(out_ptr + base + idx, out, mask=mask)


def triton_conv_transpose3d_swish(x, weight, bias, stride, padding):
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    SD = SH = SW = stride
    PD = PH = PW = padding

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_W = 32 if OW >= 32 else 16
    grid = (N * OC, OD * OH, (OW + BLOCK_W - 1) // BLOCK_W)

    conv_transpose3d_swish_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_W=BLOCK_W,
        num_warps=4,
    )
    return out


def triton_group_norm_hardswish(x, weight, bias, G, eps):
    x = x.contiguous()
    N, C = x.shape[0], x.shape[1]
    S = 1
    for d in x.shape[2:]:
        S *= d
    C_per_G = C // G
    out = torch.empty_like(x)
    group_elems = C_per_G * S

    if group_elems >= 4096:
        BLOCK = 1024
    elif group_elems >= 1024:
        BLOCK = 1024
    elif group_elems >= 512:
        BLOCK = 512
    else:
        BLOCK = 256

    grid = (N * G,)
    group_norm_hardswish_kernel[grid](
        x, weight, bias, out,
        N, C, S, G, C_per_G, eps,
        BLOCK_SIZE=BLOCK,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.groups = groups
        self.eps = eps
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous()
        # Fused ConvTranspose3d + Swish
        bias = self.conv_transpose.bias
        if bias is None:
            bias = torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)
        x = triton_conv_transpose3d_swish(
            x, self.conv_transpose.weight, bias,
            self.stride, self.padding,
        )
        # Fused GroupNorm + HardSwish
        x = triton_group_norm_hardswish(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.groups,
            self.eps,
        )
        return x