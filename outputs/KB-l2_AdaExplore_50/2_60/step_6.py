import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def convt3d_swish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    BLOCK_SP: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # program ids: (n, oc_tile, sp_tile)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    sp_total = OD * OH * OW
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < sp_total

    ow = sp_offs % OW
    oh = (sp_offs // OW) % OH
    od = sp_offs // (OW * OH)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Load bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32) + bias[None, :]

    # For each output position, iterate input positions and kernel positions that contribute.
    # od + PD = id*SD + kd  -> for each kd, id*SD = od + PD - kd, must be divisible by SD, and 0<=id<ID
    for kd in tl.static_range(0, KD):
        id_num = od + PD - kd
        id_val = id_num // SD
        id_valid = (id_num >= 0) & (id_num % SD == 0) & (id_val >= 0) & (id_val < ID)
        for kh in tl.static_range(0, KH):
            ih_num = oh + PH - kh
            ih_val = ih_num // SH
            ih_valid = (ih_num >= 0) & (ih_num % SH == 0) & (ih_val >= 0) & (ih_val < IH)
            for kw in tl.static_range(0, KW):
                iw_num = ow + PW - kw
                iw_val = iw_num // SW
                iw_valid = (iw_num >= 0) & (iw_num % SW == 0) & (iw_val >= 0) & (iw_val < IW)
                valid = id_valid & ih_valid & iw_valid & sp_mask

                # base input index for this (id, ih, iw) per sp slot
                in_sp_idx = ((id_val * IH) + ih_val) * IW + iw_val
                # contribution: sum over ic of x[n, ic, id, ih, iw] * w[ic, oc, kd, kh, kw]
                for ic in range(0, IC):
                    x_off = pid_n * (IC * ID * IH * IW) + ic * (ID * IH * IW) + in_sp_idx
                    xv = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                    # weight: [IC, OC, KD, KH, KW]
                    w_off = ic * (OC * KD * KH * KW) + oc_offs * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    wv = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                    acc += xv[:, None] * wv[None, :]

    # Apply swish: y = acc * sigmoid(acc)
    sig = tl.sigmoid(acc)
    out = acc * sig

    # store: out shape [N, OC, OD, OH, OW]
    out_off = (pid_n * OC + oc_offs[None, :]) * sp_total + sp_offs[:, None]
    store_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, out, mask=store_mask)


@triton.jit
def gn_stats_kernel(
    x_ptr, mean_ptr, rstd_ptr,
    C, G, CPG, S,
    eps,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G
    base = n * C * S + g * CPG * S

    sum_val = 0.0
    sum_sq = 0.0
    for off in range(0, GROUP_SIZE, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < GROUP_SIZE
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / GROUP_SIZE
    var = sum_sq / GROUP_SIZE - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)


@triton.jit
def gn_apply_hardswish_kernel(
    x_ptr, out_ptr, mean_ptr, rstd_ptr, weight_ptr, bias_ptr,
    C, G, CPG, S, TOTAL,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOTAL

    c_idx = (offsets // S) % C
    n_idx = offsets // (C * S)
    g_idx = c_idx // CPG
    stat_idx = n_idx * G + g_idx

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + stat_idx, mask=mask, other=0.0)
    rstd = tl.load(rstd_ptr + stat_idx, mask=mask, other=0.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=0.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    normed = (x - mean) * rstd
    y = normed * w + b
    t = y + 3.0
    t = tl.minimum(tl.maximum(t, 0.0), 6.0)
    out = y * t * (1.0 / 6.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def convt3d_swish(x, weight, bias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    SD = SH = SW = stride
    PD = PH = PW = padding

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

    BLOCK_SP = 64
    BLOCK_OC = 16
    sp_total = OD * OH * OW
    grid = (N, (OC + BLOCK_OC - 1) // BLOCK_OC, (sp_total + BLOCK_SP - 1) // BLOCK_SP)

    convt3d_swish_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_SP=BLOCK_SP,
        BLOCK_OC=BLOCK_OC,
        num_warps=4,
        num_stages=2,
    )
    return out


def gn_hardswish(x, weight, bias, groups, eps):
    x = x.contiguous()
    N, C = x.shape[0], x.shape[1]
    spatial = 1
    for s in x.shape[2:]:
        spatial *= s
    G = groups
    CPG = C // G
    group_size = CPG * spatial

    mean = torch.empty((N * G,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((N * G,), device=x.device, dtype=torch.float32)

    block = 1
    while block < group_size and block < 1024:
        block *= 2
    if block > 1024:
        block = 1024

    grid_stats = (N * G,)
    gn_stats_kernel[grid_stats](
        x, mean, rstd,
        C, G, CPG, spatial,
        eps,
        GROUP_SIZE=group_size,
        BLOCK_SIZE=block,
        num_warps=4,
    )

    out = torch.empty_like(x)
    total = N * C * spatial
    BLOCK = 1024
    grid = ((total + BLOCK - 1) // BLOCK,)
    gn_apply_hardswish_kernel[grid](
        x, out, mean, rstd, weight, bias,
        C, G, CPG, spatial, total,
        BLOCK_SIZE=BLOCK,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.groups = groups
        self.eps = eps

    def forward(self, x):
        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()
        b = self.conv_transpose.bias
        if b is None:
            b = torch.zeros(w.shape[1], device=x.device, dtype=x.dtype)
        b = b.contiguous()
        x = convt3d_swish(x, w, b, self.stride, self.padding)
        x = gn_hardswish(x, self.group_norm.weight, self.group_norm.bias, self.groups, self.eps)
        return x