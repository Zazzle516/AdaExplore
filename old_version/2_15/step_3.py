import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    BLOCK: tl.constexpr,
):
    # one program per (n, oc, od) and we iterate over (oh, ow) chunks
    pid = tl.program_id(0)
    pid_spatial = tl.program_id(1)

    n = pid // OC
    oc = pid % OC

    spatial_size = OD * OH * OW
    offs = pid_spatial * BLOCK + tl.arange(0, BLOCK)
    mask = offs < spatial_size

    od = offs // (OH * OW)
    rem = offs % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    # initialize with bias
    bias_val = tl.load(b_ptr + oc)
    acc = tl.zeros((BLOCK,), dtype=tl.float32) + bias_val

    # For ConvTranspose3d: out[n,oc,od,oh,ow] = sum over ic, kd, kh, kw of
    # x[n,ic, (od+PD-kd)/SD, (oh+PH-kh)/SH, (ow+PW-kw)/SW] * w[ic,oc,kd,kh,kw]
    # only when (od+PD-kd) is divisible by SD etc and in range

    for ic in range(0, IC):
        for kd in range(0, KD):
            id_num = od + PD - kd
            id_val = id_num // SD
            id_valid = (id_num >= 0) & ((id_num % SD) == 0) & (id_val < ID) & (id_val >= 0)
            for kh in range(0, KH):
                ih_num = oh + PH - kh
                ih_val = ih_num // SH
                ih_valid = (ih_num >= 0) & ((ih_num % SH) == 0) & (ih_val < IH) & (ih_val >= 0)
                for kw in range(0, KW):
                    iw_num = ow + PW - kw
                    iw_val = iw_num // SW
                    iw_valid = (iw_num >= 0) & ((iw_num % SW) == 0) & (iw_val < IW) & (iw_val >= 0)

                    valid = id_valid & ih_valid & iw_valid & mask

                    x_offset = ((n * IC + ic) * ID + id_val) * IH * IW + ih_val * IW + iw_val
                    w_offset = ((ic * OC + oc) * KD + kd) * KH * KW + kh * KW + kw

                    x_val = tl.load(x_ptr + x_offset, mask=valid, other=0.0)
                    w_val = tl.load(w_ptr + w_offset)

                    acc += x_val * w_val

    out_offset = ((n * OC + oc) * OD) * OH * OW + offs
    tl.store(out_ptr + out_offset, acc, mask=mask)


def conv_transpose3d_triton(x, weight, bias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    SD = SH = SW = stride
    PD = PH = PW = padding

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

    BLOCK = 128
    spatial_size = OD * OH * OW
    grid = (N * OC, (spatial_size + BLOCK - 1) // BLOCK)

    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out


@triton.jit
def bn_meansub_kernel(
    x_ptr, out_ptr,
    scale_ptr, shift_ptr,
    N, C, SPATIAL,
    BLOCK: tl.constexpr,
):
    # one program per (n, c), reduces over spatial
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    scale = tl.load(scale_ptr + c)
    shift = tl.load(shift_ptr + c)

    base = (n * C + c) * SPATIAL

    # first pass: compute sum of normalized values
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for off in range(0, SPATIAL, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < SPATIAL
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        y = x * scale + shift
        acc += tl.where(mask, y, 0.0)

    total = tl.sum(acc, axis=0)
    mean = total / SPATIAL

    # second pass: write out (normalized - mean)
    for off in range(0, SPATIAL, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < SPATIAL
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        y = x * scale + shift - mean
        tl.store(out_ptr + base + idx, y, mask=mask)


def bn_meansub_triton(x, scale, shift):
    N, C, D, H, W = x.shape
    SPATIAL = D * H * W
    out = torch.empty_like(x)
    BLOCK = 1024
    grid = (N * C,)
    bn_meansub_kernel[grid](
        x, out, scale, shift,
        N, C, SPATIAL,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous()
        if self.conv_transpose.bias is not None:
            bias = self.conv_transpose.bias.contiguous()
        else:
            bias = torch.zeros(weight.shape[1], device=x.device, dtype=x.dtype)

        y = conv_transpose3d_triton(x, weight, bias, self.stride, self.padding)

        # BatchNorm3d in training mode: compute batch stats
        # We compute mean/var over (N, D, H, W) per channel
        N, C, D, H, W = y.shape
        # Use PyTorch's batch_norm for correctness (matches training stats update)
        y = self.batch_norm(y)

        # Now subtract mean over spatial dims (2,3,4)
        # We'll do this with our kernel that just does mean subtraction
        # Since BN was already applied, scale=1, shift=0
        scale = torch.ones(C, device=y.device, dtype=y.dtype)
        shift = torch.zeros(C, device=y.device, dtype=y.dtype)
        y = bn_meansub_triton(y.contiguous(), scale, shift)
        return y