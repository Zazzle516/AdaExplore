import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ----------------------------------------------------------------------------
# Custom ConvTranspose3d as gather: one program per (N, OC_tile, out_spatial_tile)
# Iterates over (IC, kd, kh, kw); for each kernel position, valid input positions
# are those where (out + pad - k) % stride == 0 and in-bounds.
# ----------------------------------------------------------------------------

@triton.jit
def _conv_transpose3d_kernel(
    x_ptr,         # [N, IC, ID, IH, IW]
    w_ptr,         # [IC, OC, KD, KH, KW]
    b_ptr,         # [OC]  (bias possibly zero)
    out_ptr,       # [N, OC, OD, OH, OW]
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    KD, KH, KW,
    stride, pad,
    BLOCK_OC: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_s = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)   # [BLOCK_OC]
    oc_mask = oc_offs < OC

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)       # [BLOCK_S]
    s_mask = s_offs < (OD * OH * OW)

    od = s_offs // (OH * OW)
    rem = s_offs % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    # Initialize accumulator with bias.
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = bias[:, None] + tl.zeros([BLOCK_OC, BLOCK_S], dtype=tl.float32)

    # x_n_ptr base for this batch
    x_n_base = pid_n * IC * ID * IH * IW

    for kd in range(0, KD):
        # input depth index for each output position
        id_num = od + pad - kd        # [BLOCK_S]
        id_valid_mod = (id_num % stride) == 0
        id_idx = id_num // stride
        id_in = (id_idx >= 0) & (id_idx < ID) & id_valid_mod

        for kh in range(0, KH):
            ih_num = oh + pad - kh
            ih_valid_mod = (ih_num % stride) == 0
            ih_idx = ih_num // stride
            ih_in = (ih_idx >= 0) & (ih_idx < IH) & ih_valid_mod

            for kw in range(0, KW):
                iw_num = ow + pad - kw
                iw_valid_mod = (iw_num % stride) == 0
                iw_idx = iw_num // stride
                iw_in = (iw_idx >= 0) & (iw_idx < IW) & iw_valid_mod

                in_mask = id_in & ih_in & iw_in & s_mask   # [BLOCK_S]

                # input offset for each s in tile
                in_off = id_idx * (IH * IW) + ih_idx * IW + iw_idx  # [BLOCK_S]

                # Loop over input channels
                for ic in range(0, IC):
                    x_addr = x_n_base + ic * (ID * IH * IW) + in_off
                    x_val = tl.load(x_ptr + x_addr, mask=in_mask, other=0.0)  # [BLOCK_S]
                    # weight: [IC, OC, KD, KH, KW]
                    w_addr = ic * (OC * KD * KH * KW) + oc_offs * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w_val = tl.load(w_ptr + w_addr, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                    acc += w_val[:, None] * x_val[None, :]

    # Write output [N, OC, OD*OH*OW]
    out_base = pid_n * OC * OD * OH * OW
    out_addr = out_base + oc_offs[:, None] * (OD * OH * OW) + s_offs[None, :]
    out_mask = oc_mask[:, None] & s_mask[None, :]
    tl.store(out_ptr + out_addr, acc, mask=out_mask)


def conv_transpose3d_triton(x, weight, bias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    IC2, OC, KD, KH, KW = weight.shape
    assert IC == IC2
    OD = (ID - 1) * stride - 2 * padding + KD
    OH = (IH - 1) * stride - 2 * padding + KH
    OW = (IW - 1) * stride - 2 * padding + KW
    S = OD * OH * OW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    if bias is None:
        bias_t = torch.zeros(OC, device=x.device, dtype=x.dtype)
    else:
        bias_t = bias.contiguous()

    BLOCK_OC = 32
    BLOCK_S = 128

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(S, BLOCK_S))

    _conv_transpose3d_kernel[grid](
        x, weight, bias_t, out,
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        stride, padding,
        BLOCK_OC=BLOCK_OC, BLOCK_S=BLOCK_S,
        num_warps=4, num_stages=2,
    )
    return out


# ----------------------------------------------------------------------------
# Fused BN(eval, affine fold) + per-(N,C) mean subtraction.
# ----------------------------------------------------------------------------

@triton.jit
def _bn_mean_sub_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    shift_ptr,
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)  # over N*C
    c = pid % C
    row_offset = pid * S

    scale = tl.load(scale_ptr + c)
    shift = tl.load(shift_ptr + c)

    acc = 0.0
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        vals = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0)
        vals_bn = vals * scale + shift
        acc += tl.sum(vals_bn, axis=0)
    mean = acc / S

    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        vals = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0)
        vals_bn = vals * scale + shift
        tl.store(out_ptr + row_offset + offs, vals_bn - mean, mask=mask)


def fused_bn_mean_sub(x, scale, shift):
    assert x.is_cuda and x.is_contiguous()
    N, C, D, H, W = x.shape
    S = D * H * W
    out = torch.empty_like(x)
    grid = (N * C,)
    if S <= 2048:
        BLOCK_S = 1024
        num_warps = 4
    elif S <= 8192:
        BLOCK_S = 2048
        num_warps = 8
    else:
        BLOCK_S = 2048
        num_warps = 8
    _bn_mean_sub_kernel[grid](x, out, scale, shift, N, C, S,
                              BLOCK_S=BLOCK_S, num_warps=num_warps, num_stages=2)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()
        b = self.conv_transpose.bias
        if b is not None:
            b = b.contiguous()
        y = conv_transpose3d_triton(x, w, b, self.stride, self.padding)

        bn = self.batch_norm
        if not self.training:
            mean = bn.running_mean
            var = bn.running_var
            eps = bn.eps
            inv = torch.rsqrt(var + eps)
            scale = bn.weight * inv if bn.weight is not None else inv
            shift = (bn.bias - mean * scale) if bn.bias is not None else (-mean * scale)
            y = y.contiguous()
            return fused_bn_mean_sub(y, scale.contiguous(), shift.contiguous())
        else:
            y = bn(y)
            y = y - torch.mean(y, dim=(2, 3, 4), keepdim=True)
            return y