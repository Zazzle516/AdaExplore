import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SP': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 256}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'OC', 'OD', 'OH', 'OW'],
)
@triton.jit
def conv_transpose3d_swish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, ID, IH, IW,
    OC: tl.constexpr, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_sp = tl.program_id(0)
    pid_n = tl.program_id(1)

    num_ow_tiles = (OW + BLOCK_SP - 1) // BLOCK_SP
    ow_tile = pid_sp % num_ow_tiles
    tmp = pid_sp // num_ow_tiles
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    n = pid_n

    ow_offs = ow_tile * BLOCK_SP + tl.arange(0, BLOCK_SP)
    ow_mask = ow_offs < OW

    oc_offs = tl.arange(0, OC)

    acc = tl.zeros([OC, BLOCK_SP], dtype=tl.float32)

    for kd in tl.static_range(KD):
        id_num = od + PAD - kd
        id_div = id_num // STRIDE
        id_rem = id_num - id_div * STRIDE
        id_valid = (id_rem == 0) & (id_div >= 0) & (id_div < ID)
        for kh in tl.static_range(KH):
            ih_num = oh + PAD - kh
            ih_div = ih_num // STRIDE
            ih_rem = ih_num - ih_div * STRIDE
            ih_valid = (ih_rem == 0) & (ih_div >= 0) & (ih_div < IH)
            dh_valid = id_valid & ih_valid
            for kw in tl.static_range(KW):
                iw_num = ow_offs + PAD - kw
                iw_div = iw_num // STRIDE
                iw_rem = iw_num - iw_div * STRIDE
                iw_valid = (iw_rem == 0) & (iw_div >= 0) & (iw_div < IW) & ow_mask
                valid = iw_valid & dh_valid

                for ic in tl.static_range(IC):
                    w_off = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off)
                    x_off = ((n * IC + ic) * ID + id_div) * IH * IW + ih_div * IW + iw_div
                    x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                    acc += w_val[:, None] * x_val[None, :]

    bias = tl.load(b_ptr + oc_offs)
    y = acc + bias[:, None]
    y = y * tl.sigmoid(y)

    out_off = ((n * OC + oc_offs[:, None]) * OD + od) * OH * OW + oh * OW + ow_offs[None, :]
    out_mask = ow_mask[None, :]
    tl.store(out_ptr + out_off, y, mask=out_mask)


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

    inv_n = 1.0 / group_elems
    mean = sum_val * inv_n
    var = sum_sq * inv_n - mean * mean
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
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    OD = (ID - 1) * stride - 2 * padding + KD
    OH = (IH - 1) * stride - 2 * padding + KH
    OW = (IW - 1) * stride - 2 * padding + KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        OD * OH * ((OW + META['BLOCK_SP'] - 1) // META['BLOCK_SP']),
        N,
    )
    conv_transpose3d_swish_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        stride, padding,
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
    if group_elems >= 16384:
        BLOCK = 4096
        nw = 8
        ns = 3
    elif group_elems >= 4096:
        BLOCK = 2048
        nw = 8
        ns = 3
    elif group_elems >= 1024:
        BLOCK = 1024
        nw = 8
        ns = 2
    elif group_elems >= 512:
        BLOCK = 512
        nw = 4
        ns = 2
    elif group_elems >= 256:
        BLOCK = 256
        nw = 4
        ns = 2
    else:
        BLOCK = 128
        nw = 4
        ns = 2
    grid = (N * G,)
    group_norm_hardswish_kernel[grid](
        x, weight, bias, out,
        N, C, S, G, C_per_G, eps,
        BLOCK_SIZE=BLOCK,
        num_warps=nw,
        num_stages=ns,
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
        self.has_bias = bias

    def forward(self, x):
        x = x.contiguous()
        if self.has_bias:
            bias = self.conv_transpose.bias
        else:
            bias = torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)
        x = triton_conv_transpose3d_swish(
            x,
            self.conv_transpose.weight,
            bias,
            self.stride,
            self.padding,
        )
        x = triton_group_norm_hardswish(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.groups,
            self.eps,
        )
        return x