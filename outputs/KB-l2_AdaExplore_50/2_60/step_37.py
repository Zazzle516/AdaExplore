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
    BLOCK_OC: tl.constexpr, BLOCK_SP: tl.constexpr,
):
    # grid: (N, ceil(OC/BLOCK_OC), ceil(OD*OH*OW / BLOCK_SP))
    n = tl.program_id(0)
    oc_block = tl.program_id(1)
    sp_block = tl.program_id(2)

    oc_offs = oc_block * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    sp_offs = sp_block * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]

    OHW = OH * OW
    OSP = OD * OH * OW

    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < OSP

    # initialize with bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc = bias[:, None] + tl.zeros([BLOCK_OC, BLOCK_SP], dtype=tl.float32)

    # For each (kd, kh, kw), determine input position
    # output[o] = sum over k where (o + pad - k) % stride == 0 and id = (o+pad-k)/stride in [0, ID)
    for kd in tl.static_range(0, KD):
        id_num = od + PAD - kd
        id_valid = (id_num % STRIDE) == 0
        id_ = id_num // STRIDE
        id_in = (id_ >= 0) & (id_ < ID) & id_valid  # [BLOCK_SP]
        for kh in tl.static_range(0, KH):
            ih_num = oh + PAD - kh
            ih_valid = (ih_num % STRIDE) == 0
            ih_ = ih_num // STRIDE
            ih_in = (ih_ >= 0) & (ih_ < IH) & ih_valid
            for kw in tl.static_range(0, KW):
                iw_num = ow + PAD - kw
                iw_valid = (iw_num % STRIDE) == 0
                iw_ = iw_num // STRIDE
                iw_in = (iw_ >= 0) & (iw_ < IW) & iw_valid

                spatial_valid = id_in & ih_in & iw_in  # [BLOCK_SP]
                in_sp_offset = id_ * (IH * IW) + ih_ * IW + iw_  # [BLOCK_SP]

                # load over IC, accumulate
                for ic in range(0, IC):
                    # x[n, ic, id_, ih_, iw_]
                    x_addr = (n * IC + ic) * (ID * IH * IW) + in_sp_offset
                    x_val = tl.load(x_ptr + x_addr, mask=spatial_valid & sp_mask, other=0.0)  # [BLOCK_SP]

                    # w[ic, oc, kd, kh, kw]
                    w_addr = ic * (OC * KD * KH * KW) + oc_offs * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w_val = tl.load(w_ptr + w_addr, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    acc += w_val[:, None] * x_val[None, :]

    # Swish
    sig = tl.sigmoid(acc)
    out = acc * sig

    # store: out[n, oc, od, oh, ow] -- channels first layout
    # out_addr[oc, sp] = ((n * OC + oc) * OSP) + sp
    out_addr = (n * OC + oc_offs)[:, None] * OSP + sp_offs[None, :]
    store_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_addr, out, mask=store_mask)


@triton.jit
def group_norm_hardswish_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C, S, G,
    C_per_G: tl.constexpr,
    eps,
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

    OSP = OD * OH * OW
    BLOCK_OC = 16
    BLOCK_SP = 128

    grid = (N, (OC + BLOCK_OC - 1) // BLOCK_OC, (OSP + BLOCK_SP - 1) // BLOCK_SP)

    conv_transpose3d_swish_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        stride, padding,
        BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
        num_warps=4, num_stages=2,
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
        N, C, S, G,
        C_per_G,
        eps,
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
        x = x.cuda()
        # fused conv_transpose3d + swish
        x = triton_conv_transpose3d_swish(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.stride,
            self.padding,
        )
        # fused group_norm + hardswish
        x = triton_group_norm_hardswish(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.groups,
            self.eps,
        )
        return x