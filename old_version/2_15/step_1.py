import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    # program_id(0): n
    # program_id(1): oc tile
    # program_id(2): spatial tile (flattened OD*OH*OW)
    n = tl.program_id(0)
    oc_block = tl.program_id(1)
    sp_block = tl.program_id(2)

    oc_offs = oc_block * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = sp_block * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (OD * OH * OW)

    # decompose sp_offs -> (od, oh, ow)
    od = sp_offs // (OH * OW)
    rem = sp_offs % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    # init with bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32) + bias[:, None]

    # For ConvTranspose3d:
    # out[n,oc,od,oh,ow] = sum_{ic,kd,kh,kw} x[n,ic,id,ih,iw] * w[ic,oc,kd,kh,kw]
    # where id*SD - PD + kd = od => id = (od + PD - kd) / SD, valid if divisible and in range
    for kd in range(KD):
        id_num = od + PD - kd
        id_val = id_num // SD
        id_valid = (id_num >= 0) & (id_num % SD == 0) & (id_val >= 0) & (id_val < ID)
        for kh in range(KH):
            ih_num = oh + PH - kh
            ih_val = ih_num // SH
            ih_valid = (ih_num >= 0) & (ih_num % SH == 0) & (ih_val >= 0) & (ih_val < IH)
            for kw in range(KW):
                iw_num = ow + PW - kw
                iw_val = iw_num // SW
                iw_valid = (iw_num >= 0) & (iw_num % SW == 0) & (iw_val >= 0) & (iw_val < IW)

                spatial_valid = id_valid & ih_valid & iw_valid & sp_mask

                # input index base: n*IC*ID*IH*IW + ic*ID*IH*IW + id*IH*IW + ih*IW + iw
                in_spatial_idx = id_val * (IH * IW) + ih_val * IW + iw_val
                # weight index base: ic*OC*KD*KH*KW + oc*KD*KH*KW + kd*KH*KW + kh*KW + kw
                w_kern_idx = kd * (KH * KW) + kh * KW + kw

                for ic in range(IC):
                    x_idx = n * (IC * ID * IH * IW) + ic * (ID * IH * IW) + in_spatial_idx
                    x_vals = tl.load(x_ptr + x_idx, mask=spatial_valid, other=0.0)  # [BLOCK_SP]
                    w_idx = ic * (OC * KD * KH * KW) + oc_offs * (KD * KH * KW) + w_kern_idx
                    w_vals = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                    acc += w_vals[:, None] * x_vals[None, :]

    # store
    out_idx = n * (OC * OD * OH * OW) + oc_offs[:, None] * (OD * OH * OW) + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


def conv_transpose3d_triton(x, weight, bias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    assert IC == IC_w
    SD, SH, SW = stride
    PD, PH, PW = padding

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

    BLOCK_OC = 32
    BLOCK_SP = 64

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OD * OH * OW, BLOCK_SP))

    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_OC=BLOCK_OC,
        BLOCK_SP=BLOCK_SP,
        num_warps=4,
        num_stages=2,
    )
    return out


@triton.jit
def bn_meansub_kernel(
    x_ptr, out_ptr,
    scale_ptr, shift_ptr,
    N, C, SP,
    BLOCK_SP: tl.constexpr,
):
    # one program per (n, c), reduces over SP elements
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    scale = tl.load(scale_ptr + c)
    shift = tl.load(shift_ptr + c)

    base = n * C * SP + c * SP

    # first pass: compute sum of BN output
    sum_val = 0.0
    for off in range(0, SP, BLOCK_SP):
        idx = off + tl.arange(0, BLOCK_SP)
        mask = idx < SP
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        y = x * scale + shift
        sum_val += tl.sum(tl.where(mask, y, 0.0), axis=0)

    mean = sum_val / SP

    # second pass: write BN output minus mean
    for off in range(0, SP, BLOCK_SP):
        idx = off + tl.arange(0, BLOCK_SP)
        mask = idx < SP
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        y = x * scale + shift - mean
        tl.store(out_ptr + base + idx, y, mask=mask)


def bn_meansub_triton(x, scale, shift):
    N, C, D, H, W = x.shape
    SP = D * H * W
    out = torch.empty_like(x)
    grid = (N * C,)
    BLOCK_SP = 1024
    bn_meansub_kernel[grid](
        x, out, scale, shift,
        N, C, SP,
        BLOCK_SP=BLOCK_SP,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        if isinstance(stride, int):
            self.stride = (stride, stride, stride)
        else:
            self.stride = tuple(stride)
        if isinstance(padding, int):
            self.padding = (padding, padding, padding)
        else:
            self.padding = tuple(padding)

    def forward(self, x):
        x = x.cuda().contiguous().float()
        weight = self.conv_transpose.weight.cuda().contiguous().float()
        if self.conv_transpose.bias is not None:
            bias = self.conv_transpose.bias.cuda().contiguous().float()
        else:
            bias = torch.zeros(self.out_channels, device=x.device, dtype=torch.float32)

        y = conv_transpose3d_triton(x, weight, bias, self.stride, self.padding)

        # BN params
        bn = self.batch_norm
        if bn.training:
            # need to compute running stats; do BN via PyTorch to keep training-mode semantics
            y = bn(y)
            y = y - torch.mean(y, dim=(2, 3, 4), keepdim=True)
            return y
        else:
            running_mean = bn.running_mean
            running_var = bn.running_var
            eps = bn.eps
            w = bn.weight if bn.weight is not None else torch.ones_like(running_mean)
            b = bn.bias if bn.bias is not None else torch.zeros_like(running_mean)
            invstd = torch.rsqrt(running_var + eps)
            scale = (w * invstd).contiguous()
            shift = (b - running_mean * w * invstd).contiguous()
            y = bn_meansub_triton(y, scale, shift)
            return y