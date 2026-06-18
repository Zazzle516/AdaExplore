import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, cb_ptr, scale_ptr, bias_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    OH_OW: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc_blk = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    S = OD * OH * OW
    mask_s = offs_s < S

    offs_oc = pid_oc_blk * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    od = offs_s // OH_OW
    rem = offs_s % OH_OW
    oh = rem // OW
    ow = rem % OW

    # load conv bias for OC tile
    conv_b = tl.load(cb_ptr + offs_oc, mask=mask_oc, other=0.0)  # [BLOCK_OC]

    # accumulator [BLOCK_S, BLOCK_OC]
    acc = tl.zeros((BLOCK_S, BLOCK_OC), dtype=tl.float32) + conv_b[None, :]

    IH_IW = IH * IW
    KH_KW = KH * KW

    for ic in range(0, IC):
        for kd in tl.static_range(0, KD):
            id_ = od + kd
            for kh in tl.static_range(0, KH):
                ih = oh + kh
                for kw in tl.static_range(0, KW):
                    iw = ow + kw
                    in_off = ((pid_n * IC + ic) * ID + id_) * IH_IW + ih * IW + iw
                    x_val = tl.load(x_ptr + in_off, mask=mask_s, other=0.0)  # [BLOCK_S]

                    w_off = ((offs_oc * IC + ic) * KD + kd) * KH_KW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)  # [BLOCK_OC]

                    acc += x_val[:, None] * w_val[None, :]

    # Epilogue per OC
    sc = tl.load(scale_ptr + offs_oc, mask=mask_oc, other=0.0)  # [BLOCK_OC]
    bi = tl.load(bias_ptr + offs_oc, mask=mask_oc, other=0.0)   # [BLOCK_OC]

    y = acc * sc[None, :]
    e2 = tl.exp(2.0 * y)
    t = (e2 - 1.0) / (e2 + 1.0)
    z = t * bi[None, :]
    out = 1.0 / (1.0 + tl.exp(-z))

    # Store: out shape (N, OC, S). offset = (pid_n * OC + oc) * S + s
    out_off = (pid_n * OC + offs_oc[None, :]) * S + offs_s[:, None]
    mask_out = mask_s[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_off, out, mask=mask_out)


def conv3d_fused(x, weight, conv_bias, scale, bias):
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1
    S = OD * OH * OW

    x_c = x.contiguous()
    w_c = weight.contiguous()
    cb_c = conv_bias.contiguous()
    scale_flat = scale.contiguous().view(-1)
    bias_flat = bias.contiguous().view(-1)

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_S = 128
    BLOCK_OC = 16
    grid = (N, (OC + BLOCK_OC - 1) // BLOCK_OC, (S + BLOCK_S - 1) // BLOCK_S)

    conv3d_fused_kernel[grid](
        x_c, w_c, cb_c, scale_flat, bias_flat, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        OH_OW=OH * OW,
        BLOCK_S=BLOCK_S,
        BLOCK_OC=BLOCK_OC,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        return conv3d_fused(
            x,
            self.conv.weight,
            self.conv.bias,
            self.scaling_factor,
            self.bias,
        )