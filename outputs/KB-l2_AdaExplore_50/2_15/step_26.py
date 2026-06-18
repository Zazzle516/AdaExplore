import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _convt3d_scatter_kernel(
    x_ptr,        # [N, IC, ID, IH, IW]
    w_ptr,        # [IC, OC, KD, KH, KW]
    out_ptr,      # [N, OC, OD, OH, OW]
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    KD, KH, KW,
    STRIDE_D, STRIDE_H, STRIDE_W,
    PAD_D, PAD_H, PAD_W,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    # program ids: (n*IC, sp_block, oc_block)
    pid_nic = tl.program_id(0)
    pid_sp = tl.program_id(1)
    pid_oc = tl.program_id(2)

    n = pid_nic // IC
    ic = pid_nic % IC

    SP = ID * IH * IW
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < SP

    # decompose sp_offs into (id, ih, iw)
    iw_idx = sp_offs % IW
    tmp = sp_offs // IW
    ih_idx = tmp % IH
    id_idx = tmp // IH

    # load x[n, ic, id, ih, iw]
    x_idx = ((n * IC + ic) * ID + id_idx) * IH * IW + ih_idx * IW + iw_idx
    x_vals = tl.load(x_ptr + x_idx, mask=sp_mask, other=0.0)  # [BLOCK_SP]

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # iterate kernel positions
    for kd in range(0, KD):
        od_idx = id_idx * STRIDE_D - PAD_D + kd  # [BLOCK_SP]
        od_valid = (od_idx >= 0) & (od_idx < OD)
        for kh in range(0, KH):
            oh_idx = ih_idx * STRIDE_H - PAD_H + kh
            oh_valid = (oh_idx >= 0) & (oh_idx < OH)
            for kw in range(0, KW):
                ow_idx = iw_idx * STRIDE_W - PAD_W + kw
                ow_valid = (ow_idx >= 0) & (ow_idx < OW)

                spatial_valid = od_valid & oh_valid & ow_valid & sp_mask

                # load weight [OC] for this (ic, kd, kh, kw)
                w_base = ic * OC * KD * KH * KW + oc_offs * (KD * KH * KW) + kd * KH * KW + kh * KW + kw
                w_vals = tl.load(w_ptr + w_base, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                # outer product: [BLOCK_SP, BLOCK_OC]
                contrib = x_vals[:, None] * w_vals[None, :]

                # output offset
                out_idx = (((n * OC + oc_offs[None, :]) * OD + od_idx[:, None]) * OH + oh_idx[:, None]) * OW + ow_idx[:, None]
                mask2d = spatial_valid[:, None] & oc_mask[None, :]
                tl.atomic_add(out_ptr + out_idx, contrib, mask=mask2d)


@triton.jit
def _bn_sub_mean_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    shift_ptr,
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    base = (n * C + c) * S

    scale = tl.load(scale_ptr + c)
    shift = tl.load(shift_ptr + c)

    acc = 0.0
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = x * scale + shift
        acc += tl.sum(tl.where(mask, y, 0.0), axis=0)
    mean = acc / S

    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = x * scale + shift - mean
        tl.store(out_ptr + base + offs, y, mask=mask)


def _convt3d_triton(x, weight, bias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    sD, sH, sW = stride, stride, stride
    pD, pH, pW = padding, padding, padding

    OD = (ID - 1) * sD - 2 * pD + KD
    OH = (IH - 1) * sH - 2 * pH + KH
    OW = (IW - 1) * sW - 2 * pW + KW

    if bias is not None:
        out = bias.view(1, OC, 1, 1, 1).expand(N, OC, OD, OH, OW).contiguous()
    else:
        out = torch.zeros(N, OC, OD, OH, OW, device=x.device, dtype=x.dtype)

    BLOCK_SP = 64
    BLOCK_OC = 32
    SP = ID * IH * IW
    grid = (N * IC, (SP + BLOCK_SP - 1) // BLOCK_SP, (OC + BLOCK_OC - 1) // BLOCK_OC)

    _convt3d_scatter_kernel[grid](
        x, weight, out,
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        sD, sH, sW,
        pD, pH, pW,
        BLOCK_OC=BLOCK_OC,
        BLOCK_SP=BLOCK_SP,
        num_warps=4,
        num_stages=2,
    )
    return out


def bn_sub_spatial_mean(x, scale, shift):
    N, C, D, H, W = x.shape
    S = D * H * W
    x = x.contiguous()
    out = torch.empty_like(x)
    BLOCK_S = 1024
    grid = (N * C,)
    _bn_sub_mean_kernel[grid](x, out, scale, shift, N, C, S, BLOCK_S=BLOCK_S, num_warps=4)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = x.contiguous()
        weight = self.conv_transpose.weight.contiguous()
        bias = self.conv_transpose.bias.contiguous() if self.conv_transpose.bias is not None else None

        if not self.training:
            y = _convt3d_triton(x, weight, bias, self.stride, self.padding)
            bn = self.batch_norm
            eps = bn.eps
            inv = torch.rsqrt(bn.running_var + eps)
            scale = (bn.weight * inv).contiguous()
            shift = (bn.bias - bn.running_mean * scale).contiguous()
            return bn_sub_spatial_mean(y, scale, shift)
        else:
            x = self.conv_transpose(x)
            x = self.batch_norm(x)
            x = x - torch.mean(x, dim=(2, 3, 4), keepdim=True)
            return x