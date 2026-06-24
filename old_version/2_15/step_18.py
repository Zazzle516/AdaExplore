import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_scatter_kernel(
    x_ptr, w_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # program_id(0): n * (ID*IH*IW) flattened input spatial
    # program_id(1): oc tile
    pid_nsp = tl.program_id(0)
    pid_oc = tl.program_id(1)

    ISP = ID * IH * IW
    n = pid_nsp // ISP
    sp = pid_nsp % ISP
    id_ = sp // (IH * IW)
    rem = sp % (IH * IW)
    ih = rem // IW
    iw = rem % IW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Load input values for all IC at this (n, id, ih, iw)
    # Loop over ic in tiles, accumulate scatter contributions for this (n, sp)
    # Then scatter-add to each (kd, kh, kw)

    # accumulator across IC for each (kd,kh,kw, oc): we compute on the fly
    # For each (kd,kh,kw), compute output position
    for kd in range(KD):
        od = id_ * SD - PD + kd
        d_valid = (od >= 0) & (od < OD)
        for kh in range(KH):
            oh = ih * SH - PH + kh
            h_valid = (oh >= 0) & (oh < OH)
            for kw in range(KW):
                ow = iw * SW - PW + kw
                w_valid = (ow >= 0) & (ow < OW)
                valid = d_valid & h_valid & w_valid

                # Compute sum over IC of x[n,ic,id,ih,iw] * w[ic,oc,kd,kh,kw]
                acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)
                for ic_start in range(0, IC, BLOCK_IC):
                    ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                    ic_mask = ic_offs < IC

                    x_idx = n * (IC * ISP) + ic_offs * ISP + sp
                    x_vals = tl.load(x_ptr + x_idx, mask=ic_mask, other=0.0)  # [BLOCK_IC]

                    w_idx = ic_offs[:, None] * (OC * KD * KH * KW) + \
                            oc_offs[None, :] * (KD * KH * KW) + \
                            kd * (KH * KW) + kh * KW + kw
                    w_mask = ic_mask[:, None] & oc_mask[None, :]
                    w_vals = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0)  # [BLOCK_IC, BLOCK_OC]

                    acc += tl.sum(x_vals[:, None] * w_vals, axis=0)

                # scatter add to output
                out_idx = n * (OC * OD * OH * OW) + oc_offs * (OD * OH * OW) + \
                          od * (OH * OW) + oh * OW + ow
                store_mask = oc_mask & valid
                tl.atomic_add(out_ptr + out_idx, acc, mask=store_mask)


def conv_transpose3d_triton(x, weight, bias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    SD, SH, SW = stride
    PD, PH, PW = padding

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    # Initialize with bias broadcast
    if bias is not None:
        out = bias.view(1, OC, 1, 1, 1).expand(N, OC, OD, OH, OW).contiguous()
    else:
        out = torch.zeros((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

    BLOCK_OC = 32
    BLOCK_IC = 16
    ISP = ID * IH * IW

    grid = (N * ISP, triton.cdiv(OC, BLOCK_OC))

    conv_transpose3d_scatter_kernel[grid](
        x, weight, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_OC=BLOCK_OC,
        BLOCK_IC=BLOCK_IC,
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
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    scale = tl.load(scale_ptr + c)
    shift = tl.load(shift_ptr + c)

    base = n * C * SP + c * SP

    sum_val = tl.zeros((), dtype=tl.float32)
    for off in range(0, SP, BLOCK_SP):
        idx = off + tl.arange(0, BLOCK_SP)
        mask = idx < SP
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        y = x * scale + shift
        sum_val += tl.sum(tl.where(mask, y, 0.0), axis=0)

    mean = sum_val / SP

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
            bias = None

        y = conv_transpose3d_triton(x, weight, bias, self.stride, self.padding)

        bn = self.batch_norm
        if bn.training:
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