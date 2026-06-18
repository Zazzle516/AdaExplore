import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, y_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    IC_C: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
):
    # Gather-style transposed conv (stride=1, padding=0, kernel=KDxKHxKW)
    # output[n, oc, od, oh, ow] = sum over (ic, kd, kh, kw) of
    #   x[n, ic, od-kd, oh-kh, ow-kw] * w[ic, oc, kd, kh, kw]
    # valid when 0 <= od-kd < ID etc.
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]

    OHW = OH * OW
    OS = OD * OHW

    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    sp_mask = sp_offs < OS
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # weight layout: (IC, OC, KD, KH, KW)
    # We'll iterate kd, kh, kw, ic
    for kd in tl.static_range(0, KD):
        id_ = od - kd
        d_valid = (id_ >= 0) & (id_ < ID)
        for kh in tl.static_range(0, KH):
            ih = oh - kh
            h_valid = (ih >= 0) & (ih < IH)
            for kw in tl.static_range(0, KW):
                iw = ow - kw
                w_valid = (iw >= 0) & (iw < IW)
                spatial_valid = d_valid & h_valid & w_valid & sp_mask
                # Linear input spatial index
                in_sp = id_ * (IH * IW) + ih * IW + iw  # [BLOCK_SP]
                for ic in range(0, IC_C):
                    # load weight: w[ic, oc_offs, kd, kh, kw]
                    w_offs = ic * (OC * KD * KH * KW) + oc_offs * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w_val = tl.load(w_ptr + w_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                    # load input: x[pid_n, ic, id_, ih, iw]
                    x_offs = pid_n * (IC * ID * IH * IW) + ic * (ID * IH * IW) + in_sp
                    x_val = tl.load(x_ptr + x_offs, mask=spatial_valid, other=0.0)  # [BLOCK_SP]
                    acc += w_val[:, None] * x_val[None, :]

    # store y[pid_n, oc_offs, sp_offs]
    y_offs = pid_n * (OC * OS) + oc_offs[:, None] * OS + sp_offs[None, :]
    mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(y_ptr + y_offs, acc, mask=mask)


def conv_transpose3d_triton(x, weight):
    # x: (N, IC, ID, IH, IW), weight: (IC, OC, KD, KH, KW)
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    OD = ID + KD - 1
    OH = IH + KH - 1
    OW = IW + KW - 1

    x = x.contiguous()
    weight = weight.contiguous()
    y = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    OS = OD * OH * OW
    grid = lambda meta: (N, triton.cdiv(OC, meta['BLOCK_OC']), triton.cdiv(OS, meta['BLOCK_SP']))

    conv_transpose3d_kernel[grid](
        x, weight, y,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW, OS,
        IC_C=IC,
        KD=KD, KH=KH, KW=KW,
    )
    return y


@triton.jit
def fused_relu_groupnorm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    N, C, S, G, C_PER_G,
    inv_group_size,
    eps: tl.constexpr,
    BLOCK_S: tl.constexpr,
    C_PER_G_C: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)

    sum_x = 0.0
    sum_x2 = 0.0

    base = n * C * S + g * C_PER_G_C * S

    for c_off in tl.static_range(0, C_PER_G_C):
        c_base = base + c_off * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            x = tl.load(x_ptr + c_base + offs, mask=mask, other=0.0)
            x = tl.maximum(x, 0.0)
            sum_x += tl.sum(tl.where(mask, x, 0.0))
            sum_x2 += tl.sum(tl.where(mask, x * x, 0.0))

    mean = sum_x * inv_group_size
    var = sum_x2 * inv_group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for c_off in tl.static_range(0, C_PER_G_C):
        c_idx = g * C_PER_G_C + c_off
        w = tl.load(weight_ptr + c_idx)
        b = tl.load(bias_ptr + c_idx)
        c_base = base + c_off * S
        scale = w * rstd
        shift = b - mean * scale
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            x = tl.load(x_ptr + c_base + offs, mask=mask, other=0.0)
            x = tl.maximum(x, 0.0)
            y = x * scale + shift
            tl.store(y_ptr + c_base + offs, y, mask=mask)


def fused_relu_groupnorm(x, weight, bias, groups, eps=1e-5):
    N, C, D, H, W = x.shape
    S = D * H * W
    C_PER_G = C // groups
    x_c = x.contiguous()
    y = torch.empty_like(x_c)

    BLOCK_S = 1024
    grid = (N, groups)
    inv_group_size = 1.0 / float(C_PER_G * S)
    fused_relu_groupnorm_kernel[grid](
        x_c, y, weight, bias,
        N, C, S, groups, C_PER_G,
        inv_group_size,
        eps=eps,
        BLOCK_S=BLOCK_S,
        C_PER_G_C=C_PER_G,
        num_warps=8,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self.groups = groups
        self.eps = 1e-5
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        weight = self.conv_transpose.weight.contiguous()
        y = conv_transpose3d_triton(x, weight)
        y = fused_relu_groupnorm(
            y,
            self.group_norm.weight,
            self.group_norm.bias,
            self.groups,
            self.eps,
        )
        return y