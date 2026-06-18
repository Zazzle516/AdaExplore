import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OW': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_OW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 128}, num_warps=8, num_stages=3),
    ],
    key=['N', 'IC', 'OC', 'OD', 'OH', 'OW'],
)
@triton.jit
def conv_transpose3d_swish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, ID, IH, IW,
    OC: tl.constexpr, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    pid_spatial = tl.program_id(0)
    pid_owt = tl.program_id(1)

    oh = pid_spatial % OH
    tmp = pid_spatial // OH
    od = tmp % OD
    n = tmp // OD

    ow_offs = pid_owt * BLOCK_OW + tl.arange(0, BLOCK_OW)
    ow_mask = ow_offs < OW

    oc_offs = tl.arange(0, OC)

    acc = tl.zeros([OC, BLOCK_OW], dtype=tl.float32)

    for kd in tl.static_range(0, KD):
        id_num = od + PAD - kd
        id_val = id_num // STRIDE
        id_valid = (id_num >= 0) & ((id_num % STRIDE) == 0) & (id_val >= 0) & (id_val < ID)
        for kh in tl.static_range(0, KH):
            ih_num = oh + PAD - kh
            ih_val = ih_num // STRIDE
            ih_valid = (ih_num >= 0) & ((ih_num % STRIDE) == 0) & (ih_val >= 0) & (ih_val < IH)
            dh_valid = id_valid & ih_valid
            for kw in tl.static_range(0, KW):
                iw_num = ow_offs + PAD - kw
                iw_val = iw_num // STRIDE
                iw_valid = (iw_num >= 0) & ((iw_num % STRIDE) == 0) & (iw_val >= 0) & (iw_val < IW) & ow_mask
                if dh_valid:
                    for ic in tl.static_range(0, IC):
                        x_off = ((n * IC + ic) * ID + id_val) * IH * IW + ih_val * IW + iw_val
                        xval = tl.load(x_ptr + x_off, mask=iw_valid, other=0.0)
                        w_off = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                        wval = tl.load(w_ptr + w_off)
                        acc += wval[:, None] * xval[None, :]

    if HAS_BIAS:
        bval = tl.load(b_ptr + oc_offs)
        acc += bval[:, None]

    sig = tl.sigmoid(acc)
    res = acc * sig

    out_off = ((n * OC + oc_offs[:, None]) * OD + od) * OH * OW + oh * OW + ow_offs[None, :]
    tl.store(out_ptr + out_off, res, mask=ow_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=2),
    ],
    key=['C_per_G', 'S'],
)
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

    inv_ge = 1.0 / group_elems
    mean = sum_val * inv_ge
    var = sum_sq * inv_ge - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for start in range(0, group_elems, BLOCK_SIZE):
        idx = start + offs
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        c_in_g = idx // S
        c_global = g * C_per_G + c_in_g
        w = tl.load(weight_ptr + c_global, mask=mask, other=0.0)
        b = tl.load(bias_ptr + c_global, mask=mask, other=0.0)
        scale = rstd * w
        shift = b - mean * scale
        y = x * scale + shift
        t = y + 3.0
        t = tl.minimum(tl.maximum(t, 0.0), 6.0)
        out = y * t * 0.16666666666666666
        tl.store(out_ptr + base + idx, out, mask=mask)


def triton_group_norm_hardswish(x, weight, bias, G, eps):
    x = x.contiguous()
    N, C = x.shape[0], x.shape[1]
    S = 1
    for d in x.shape[2:]:
        S *= d
    C_per_G = C // G
    out = torch.empty_like(x)
    grid = (N * G,)
    group_norm_hardswish_kernel[grid](
        x, weight, bias, out,
        N, C, S, G, C_per_G, eps,
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

        grid = lambda META: (N * OD * OH, triton.cdiv(OW, META['BLOCK_OW']))
        conv_transpose3d_swish_kernel[grid](
            x, w, b, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            STRIDE, PAD,
            HAS_BIAS=self.has_bias,
        )

        out = triton_group_norm_hardswish(
            out,
            self.group_norm.weight,
            self.group_norm.bias,
            self.groups,
            self.eps,
        )
        return out