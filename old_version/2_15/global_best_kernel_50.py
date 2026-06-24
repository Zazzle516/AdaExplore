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
    BLOCK_SP: tl.constexpr,
):
    # program_id(0): n
    # program_id(1): oc tile
    # program_id(2): input spatial tile (flattened ID*IH*IW)
    n = tl.program_id(0)
    oc_block = tl.program_id(1)
    sp_block = tl.program_id(2)

    oc_offs = oc_block * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = sp_block * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oc_mask = oc_offs < OC
    in_sp = ID * IH * IW
    sp_mask = sp_offs < in_sp

    # decompose input spatial offset into (id, ih, iw)
    id_ = sp_offs // (IH * IW)
    rem = sp_offs % (IH * IW)
    ih_ = rem // IW
    iw_ = rem % IW

    # Load input values for all IC at this spatial location: shape (IC, BLOCK_SP)
    # We loop over IC instead.
    # For each (kd, kh, kw): out[n, oc, od, oh, ow] += sum_ic x[n,ic,id,ih,iw] * w[ic,oc,kd,kh,kw]
    # Compute weight outer product -> (BLOCK_OC, BLOCK_SP) contribution, scatter.

    for kd in range(KD):
        od = id_ * SD - PD + kd
        od_valid = (od >= 0) & (od < OD)
        for kh in range(KH):
            oh = ih_ * SH - PH + kh
            oh_valid = (oh >= 0) & (oh < OH)
            for kw in range(KW):
                ow = iw_ * SW - PW + kw
                ow_valid = (ow >= 0) & (ow < OW)
                valid = od_valid & oh_valid & ow_valid & sp_mask

                acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

                w_kern_idx = kd * (KH * KW) + kh * KW + kw

                for ic in range(IC):
                    x_idx = n * (IC * ID * IH * IW) + ic * (ID * IH * IW) + sp_offs
                    x_vals = tl.load(x_ptr + x_idx, mask=sp_mask, other=0.0)
                    w_idx = ic * (OC * KD * KH * KW) + oc_offs * (KD * KH * KW) + w_kern_idx
                    w_vals = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)
                    acc += w_vals[:, None] * x_vals[None, :]

                out_idx = (n * (OC * OD * OH * OW)
                           + oc_offs[:, None] * (OD * OH * OW)
                           + od[None, :] * (OH * OW)
                           + oh[None, :] * OW
                           + ow[None, :])
                store_mask = oc_mask[:, None] & valid[None, :]
                tl.atomic_add(out_ptr + out_idx, acc, mask=store_mask)


@triton.jit
def init_bias_kernel(out_ptr, bias_ptr, N, OC, SP, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nc = tl.program_id(1)
    n = nc // OC
    c = nc % OC
    bias = tl.load(bias_ptr + c)
    base = n * OC * SP + c * SP
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < SP
    tl.store(out_ptr + base + offs, bias, mask=mask)


def conv_transpose3d_triton(x, weight, bias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    SD, SH, SW = stride
    PD, PH, PW = padding

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)
    SP_OUT = OD * OH * OW

    # initialize with bias
    BLOCK_INIT = 512
    grid_init = (triton.cdiv(SP_OUT, BLOCK_INIT), N * OC)
    init_bias_kernel[grid_init](out, bias, N, OC, SP_OUT, BLOCK=BLOCK_INIT, num_warps=4)

    BLOCK_OC = 16
    BLOCK_SP = 128
    in_sp = ID * IH * IW
    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(in_sp, BLOCK_SP))

    conv_transpose3d_scatter_kernel[grid](
        x, weight, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_OC=BLOCK_OC,
        BLOCK_SP=BLOCK_SP,
        num_warps=8,
        num_stages=1,
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

    sum_val = 0.0
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
            bias = torch.zeros(self.out_channels, device=x.device, dtype=torch.float32)

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