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
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # grid: (N * OD * OH * OW, ceil(OC/BLOCK_OC))
    pid_spatial = tl.program_id(0)
    pid_oc = tl.program_id(1)

    ow = pid_spatial % OW
    tmp = pid_spatial // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    n = tmp // OD

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    acc = tl.zeros([BLOCK_OC], dtype=tl.float32)

    # For ConvTranspose3d:
    # out[n, oc, od, oh, ow] = sum_{ic, kd, kh, kw} x[n, ic, id, ih, iw] * w[ic, oc, kd, kh, kw]
    # where id*stride - pad + kd = od, etc.
    # so id = (od + pad - kd) / stride, must be integer and in range

    # We loop over kernel positions and check validity
    for kd in tl.static_range(0, KD):
        id_num = od + PAD - kd
        id_val = id_num // STRIDE
        id_valid = (id_num >= 0) & ((id_num % STRIDE) == 0) & (id_val >= 0) & (id_val < ID)
        for kh in tl.static_range(0, KH):
            ih_num = oh + PAD - kh
            ih_val = ih_num // STRIDE
            ih_valid = (ih_num >= 0) & ((ih_num % STRIDE) == 0) & (ih_val >= 0) & (ih_val < IH)
            for kw in tl.static_range(0, KW):
                iw_num = ow + PAD - kw
                iw_val = iw_num // STRIDE
                iw_valid = (iw_num >= 0) & ((iw_num % STRIDE) == 0) & (iw_val >= 0) & (iw_val < IW)
                valid = id_valid & ih_valid & iw_valid
                if valid:
                    # accumulate over IC
                    for ic in range(0, IC):
                        x_off = ((n * IC + ic) * ID + id_val) * IH * IW + ih_val * IW + iw_val
                        xval = tl.load(x_ptr + x_off)
                        # weight[ic, oc, kd, kh, kw]
                        w_off = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                        wval = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                        acc += xval * wval

    if HAS_BIAS:
        bval = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
        acc += bval

    # Swish: x * sigmoid(x)
    sig = tl.sigmoid(acc)
    res = acc * sig

    # Store: out[n, oc, od, oh, ow]
    out_off = ((n * OC + oc_offs) * OD + od) * OH * OW + oh * OW + ow
    tl.store(out_ptr + out_off, res, mask=oc_mask)


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
        out = y * t / 6.0
        tl.store(out_ptr + base + idx, out, mask=mask)


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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.has_bias = bias

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        STRIDE = self.stride
        PAD = self.padding
        OC = self.out_channels
        OD = (ID - 1) * STRIDE - 2 * PAD + KD
        OH = (IH - 1) * STRIDE - 2 * PAD + KH
        OW = (IW - 1) * STRIDE - 2 * PAD + KW

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        w = self.conv_transpose.weight.contiguous()
        b = self.conv_transpose.bias.contiguous() if self.has_bias else torch.empty(1, device=x.device, dtype=x.dtype)

        BLOCK_OC = 16
        if OC <= 16:
            BLOCK_OC = 16
        elif OC <= 32:
            BLOCK_OC = 32
        else:
            BLOCK_OC = 32

        grid = (N * OD * OH * OW, (OC + BLOCK_OC - 1) // BLOCK_OC)
        conv_transpose3d_swish_kernel[grid](
            x, w, b, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            STRIDE, PAD,
            BLOCK_OC=BLOCK_OC,
            HAS_BIAS=self.has_bias,
            num_warps=2,
        )

        out = triton_group_norm_hardswish(
            out,
            self.group_norm.weight,
            self.group_norm.bias,
            self.groups,
            self.eps,
        )
        return out