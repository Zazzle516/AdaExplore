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
    BLOCK_OC: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # Each program: one output element (n, od, oh, ow), tile over OC
    pid_n = tl.program_id(0)  # n
    pid_spatial = tl.program_id(1)  # od*OH*OW + oh*OW + ow
    pid_oc = tl.program_id(2)  # OC tile

    od = pid_spatial // (OH * OW)
    rem = pid_spatial % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Compute output for (n, od, oh, ow), all oc in tile
    # Output index: out[n, oc, od, oh, ow]
    # Sum over (ic, kd, kh, kw):
    #   id = (od + PD - kd) / SD  (must be integer, in [0, ID))
    #   weight: w[ic, oc, kd, kh, kw]
    # Initialize accumulator
    if HAS_BIAS:
        acc = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0).to(tl.float32)
    else:
        acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Iterate over kd, kh, kw, ic
    for kd in tl.static_range(0, KD):
        id_num = od + PD - kd
        id_q = id_num // SD
        id_r = id_num - id_q * SD
        d_valid = (id_r == 0) & (id_q >= 0) & (id_q < ID)
        for kh in tl.static_range(0, KH):
            ih_num = oh + PH - kh
            ih_q = ih_num // SH
            ih_r = ih_num - ih_q * SH
            h_valid = (ih_r == 0) & (ih_q >= 0) & (ih_q < IH)
            for kw in tl.static_range(0, KW):
                iw_num = ow + PW - kw
                iw_q = iw_num // SW
                iw_r = iw_num - iw_q * SW
                w_valid = (iw_r == 0) & (iw_q >= 0) & (iw_q < IW)
                valid = d_valid & h_valid & w_valid
                if valid:
                    # Sum over ic
                    for ic in range(0, IC):
                        x_idx = ((pid_n * IC + ic) * ID + id_q) * IH * IW + ih_q * IW + iw_q
                        x_val = tl.load(x_ptr + x_idx)
                        # weight[ic, oc, kd, kh, kw]
                        w_idx = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                        w_val = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)
                        acc += x_val * w_val

    # Apply Swish: acc * sigmoid(acc)
    sig = tl.sigmoid(acc)
    res = acc * sig

    # Store output[n, oc, od, oh, ow]
    out_idx = ((pid_n * OC + oc_offs) * OD + od) * OH * OW + oh * OW + ow
    tl.store(out_ptr + out_idx, res, mask=oc_mask)


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

    sum_val = 0.0
    sum_sq = 0.0
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
        out = y * t * (1.0 / 6.0)
        tl.store(out_ptr + base + idx, out, mask=mask)


def triton_conv_transpose3d_swish(x, weight, bias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    SD = SH = SW = stride if isinstance(stride, int) else stride[0]
    PD = PH = PW = padding if isinstance(padding, int) else padding[0]

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 16
    grid = (N, OD * OH * OW, (OC + BLOCK_OC - 1) // BLOCK_OC)

    conv_transpose3d_swish_kernel[grid](
        x, weight, bias if bias is not None else x,  # dummy if no bias
        out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_OC=BLOCK_OC,
        HAS_BIAS=(bias is not None),
        num_warps=2,
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
    if group_elems >= 1024:
        BLOCK = 1024
    elif group_elems >= 512:
        BLOCK = 512
    elif group_elems >= 256:
        BLOCK = 256
    else:
        BLOCK = 128
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

    def forward(self, x):
        x = x.contiguous()
        weight = self.conv_transpose.weight.contiguous()
        bias = self.conv_transpose.bias
        x = triton_conv_transpose3d_swish(x, weight, bias, self.stride, self.padding)
        x = triton_group_norm_hardswish(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.groups,
            self.eps,
        )
        return x