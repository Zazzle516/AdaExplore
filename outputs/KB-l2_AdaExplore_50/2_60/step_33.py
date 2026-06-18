import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_swish_kernel(
    x_ptr,           # (N, IC, D, H, W)
    w_ptr,           # (IC, OC, KD, KH, KW)
    b_ptr,           # (OC,)
    y_ptr,           # (N, OC, OD, OH, OW)
    N, IC, OC,
    D, H, W,
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_idx = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_idx < OC

    sp_idx = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_idx < (OD * OH * OW)

    od = sp_idx // (OH * OW)
    rem = sp_idx % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    # For each output location, sum contributions from input positions
    # output[n, oc, od, oh, ow] = sum_{ic, kd, kh, kw} input[n, ic, id, ih, iw] * weight[ic, oc, kd, kh, kw]
    # where id*SD - PD + kd = od  =>  id = (od + PD - kd) / SD  (must be integer)
    # similarly for ih, iw

    # initialize with bias
    bias_vals = tl.load(b_ptr + oc_idx, mask=oc_mask, other=0.0).to(tl.float32)
    acc = tl.zeros([BLOCK_SP, BLOCK_OC], dtype=tl.float32) + bias_vals[None, :]

    # Loop over kernel and input channels
    for kd in tl.static_range(0, KD):
        id_num = od + PD - kd
        id_val = id_num // SD
        id_valid = (id_num >= 0) & ((id_num % SD) == 0) & (id_val >= 0) & (id_val < D)
        for kh in tl.static_range(0, KH):
            ih_num = oh + PH - kh
            ih_val = ih_num // SH
            ih_valid = (ih_num >= 0) & ((ih_num % SH) == 0) & (ih_val >= 0) & (ih_val < H)
            for kw in tl.static_range(0, KW):
                iw_num = ow + PW - kw
                iw_val = iw_num // SW
                iw_valid = (iw_num >= 0) & ((iw_num % SW) == 0) & (iw_val >= 0) & (iw_val < W)

                spatial_valid = id_valid & ih_valid & iw_valid & sp_mask
                in_spatial_offset = id_val * (H * W) + ih_val * W + iw_val

                for ic in range(0, IC):
                    # load input scalar per sp position
                    x_off = pid_n * (IC * D * H * W) + ic * (D * H * W) + in_spatial_offset
                    x_val = tl.load(x_ptr + x_off, mask=spatial_valid, other=0.0).to(tl.float32)
                    # load weight: w[ic, oc, kd, kh, kw]
                    w_off = ic * (OC * KD * KH * KW) + oc_idx * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0).to(tl.float32)
                    acc += x_val[:, None] * w_val[None, :]

    # Swish: x * sigmoid(x)
    swish = acc * tl.sigmoid(acc)

    # Store
    out_off = (pid_n * (OC * OD * OH * OW)
               + oc_idx[None, :] * (OD * OH * OW)
               + sp_idx[:, None])
    store_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(y_ptr + out_off, swish, mask=store_mask)


def conv_transpose3d_swish(x, weight, bias, stride, padding):
    N, IC, D, H, W = x.shape
    _, OC, KD, KH, KW = weight.shape
    SD = SH = SW = stride
    PD = PH = PW = padding

    OD = (D - 1) * SD - 2 * PD + KD
    OH = (H - 1) * SH - 2 * PH + KH
    OW = (W - 1) * SW - 2 * PW + KW

    y = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 16
    BLOCK_SP = 64

    grid = (N, (OC + BLOCK_OC - 1) // BLOCK_OC, (OD * OH * OW + BLOCK_SP - 1) // BLOCK_SP)

    conv_transpose3d_swish_kernel[grid](
        x, weight, bias, y,
        N, IC, OC,
        D, H, W,
        OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_OC=BLOCK_OC,
        BLOCK_SP=BLOCK_SP,
        num_warps=4,
        num_stages=2,
    )
    return y


@triton.jit
def groupnorm_hardswish_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    N, C, S,
    G, CPG,
    eps,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    group_size = CPG * S
    base = n * C * S + g * CPG * S

    sum_val = tl.zeros([], dtype=tl.float32)
    sum_sq = tl.zeros([], dtype=tl.float32)

    offs_c = tl.arange(0, BLOCK_C)
    offs_s = tl.arange(0, BLOCK_S)

    num_s_blocks = (S + BLOCK_S - 1) // BLOCK_S

    for c_start in range(0, CPG, BLOCK_C):
        c_idx = c_start + offs_c
        c_mask = c_idx < CPG
        for sb in range(0, num_s_blocks):
            s_idx = sb * BLOCK_S + offs_s
            s_mask = s_idx < S
            mask = c_mask[:, None] & s_mask[None, :]
            ptrs = base + c_idx[:, None] * S + s_idx[None, :]
            x = tl.load(x_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
            x = tl.where(mask, x, 0.0)
            sum_val += tl.sum(x)
            sum_sq += tl.sum(x * x)

    mean = sum_val / group_size
    var = sum_sq / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for c_start in range(0, CPG, BLOCK_C):
        c_idx = c_start + offs_c
        c_mask = c_idx < CPG
        gc_idx = g * CPG + c_idx
        w = tl.load(weight_ptr + gc_idx, mask=c_mask, other=0.0).to(tl.float32)
        b = tl.load(bias_ptr + gc_idx, mask=c_mask, other=0.0).to(tl.float32)
        for sb in range(0, num_s_blocks):
            s_idx = sb * BLOCK_S + offs_s
            s_mask = s_idx < S
            mask = c_mask[:, None] & s_mask[None, :]
            ptrs = base + c_idx[:, None] * S + s_idx[None, :]
            x = tl.load(x_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
            norm = (x - mean) * rstd
            out = norm * w[:, None] + b[:, None]
            t = out + 3.0
            t = tl.minimum(tl.maximum(t, 0.0), 6.0)
            res = out * t / 6.0
            tl.store(y_ptr + ptrs, res, mask=mask)


def fused_gn_hswish(x, weight, bias, G, eps):
    N, C, D, H, W = x.shape
    S = D * H * W
    CPG = C // G
    x_flat = x.contiguous().view(N, C, S)
    y = torch.empty_like(x_flat)

    BLOCK_S = 1024
    if S < 1024:
        BLOCK_S = 1
        while BLOCK_S < S:
            BLOCK_S *= 2
        BLOCK_S = max(BLOCK_S, 16)
    BLOCK_C = 4
    if CPG < 4:
        BLOCK_C = 1
        while BLOCK_C < CPG:
            BLOCK_C *= 2

    grid = (N * G,)
    groupnorm_hardswish_kernel[grid](
        x_flat, y, weight, bias,
        N, C, S, G, CPG,
        eps,
        BLOCK_S=BLOCK_S,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return y.view(N, C, D, H, W)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.groups = groups
        self.eps = eps
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()
        b = self.conv_transpose.bias.contiguous() if self.conv_transpose.bias is not None else torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)
        x = conv_transpose3d_swish(x, w, b, self.stride, self.padding)
        x = fused_gn_hswish(x, self.group_norm.weight, self.group_norm.bias, self.groups, self.eps)
        return x