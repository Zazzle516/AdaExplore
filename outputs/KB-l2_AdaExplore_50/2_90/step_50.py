import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, b_ptr, sum_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    BLOCK_N: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    # program ids
    pid_n_oc = tl.program_id(0)
    pid_sp = tl.program_id(1)

    pid_n = pid_n_oc // (OC // BLOCK_OC)
    pid_oc = pid_n_oc % (OC // BLOCK_OC)

    oc_start = pid_oc * BLOCK_OC
    sp_start = pid_sp * BLOCK_SP

    offs_oc = oc_start + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    offs_sp = sp_start + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]

    sp_total = OD * OH * OW
    sp_mask = offs_sp < sp_total

    # decompose sp into (od, oh, ow)
    ow = offs_sp % OW
    tmp = offs_sp // OW
    oh = tmp % OH
    od = tmp // OH

    # accumulator
    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # Loop over IC * KD * KH * KW
    # weight shape: (OC, IC, KD, KH, KW)
    # input shape: (N, IC, ID, IH, IW)
    K = IC * KD * KH * KW

    for k in range(0, K):
        # decompose k
        kw = k % KW
        tmp_k = k // KW
        kh = tmp_k % KH
        tmp_k2 = tmp_k // KH
        kd = tmp_k2 % KD
        ic = tmp_k2 // KD

        id_ = od + kd
        ih = oh + kh
        iw = ow + kw

        # input offset for batch pid_n
        x_offs = pid_n * (IC * ID * IH * IW) + ic * (ID * IH * IW) + id_ * (IH * IW) + ih * IW + iw  # [BLOCK_SP]
        x_vals = tl.load(x_ptr + x_offs, mask=sp_mask, other=0.0)  # [BLOCK_SP]

        # weight offset for offs_oc
        w_offs = offs_oc * (IC * KD * KH * KW) + ic * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw  # [BLOCK_OC]
        w_vals = tl.load(w_ptr + w_offs, mask=offs_oc < OC, other=0.0)  # [BLOCK_OC]

        acc += w_vals[:, None] * x_vals[None, :]

    # add bias
    b_vals = tl.load(b_ptr + offs_oc, mask=offs_oc < OC, other=0.0)
    acc += b_vals[:, None]

    # LeakyReLU(0.2)
    acc = tl.where(acc >= 0, acc, acc * 0.2)

    # add sum_tensor (per-OC)
    s_vals = tl.load(sum_ptr + offs_oc, mask=offs_oc < OC, other=0.0)
    acc += s_vals[:, None]

    # clamp [-1, 1]
    acc = tl.minimum(tl.maximum(acc, -1.0), 1.0)

    # GELU exact
    inv_sqrt2 = 0.7071067811865475
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # store: output is (N, OC, OD, OH, OW), contiguous
    # out_offs[oc, sp] = pid_n * (OC*sp_total) + (oc_start + ioc) * sp_total + offs_sp
    out_base = pid_n * (OC * sp_total)
    out_offs = out_base + offs_oc[:, None] * sp_total + offs_sp[None, :]
    mask_out = (offs_oc[:, None] < OC) & sp_mask[None, :]
    tl.store(out_ptr + out_offs, acc, mask=mask_out)


def conv3d_fused(x, weight, bias, sum_tensor):
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 64  # full OC tile
    BLOCK_SP = 128

    sp_total = OD * OH * OW
    sum_flat = sum_tensor.view(-1).contiguous()

    grid = (
        N * (OC // BLOCK_OC),
        (sp_total + BLOCK_SP - 1) // BLOCK_SP,
    )

    conv3d_fused_kernel[grid](
        x, weight, bias, sum_flat, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        BLOCK_N=1,
        BLOCK_OC=BLOCK_OC,
        BLOCK_SP=BLOCK_SP,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x):
        x = x.cuda().contiguous()
        return conv3d_fused(x, self.conv.weight, self.conv.bias, self.sum_tensor)