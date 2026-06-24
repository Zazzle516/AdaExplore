import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_scatter_kernel(
    x_ptr, w_ptr, out_ptr,
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    BLOCK_OC: tl.constexpr,
):
    # Each program: one (n, id, ih, iw_tile) - scatter-adds to output
    # Actually we'll do: one program per (n, id, ih) iterating over iw and (kd,kh,kw)
    # Then accumulate over IC into a register tile of size BLOCK_OC for each output position
    # Better: Each program handles (n, oc_tile, od, oh_tile) - gather approach
    pass


# We use a gather-based GEMM: one program per (n, oc_block, output_spatial_block)
# For each output position, sum over (ic, kd, kh, kw) where input is valid.

@triton.jit
def conv_transpose3d_gemm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, OC,
    ID, IH, IW,
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

    spatial = OD * OH * OW
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < spatial

    od = sp_offs // (OH * OW)
    rem = sp_offs % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    bias_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32) + bias_vals[None, :]

    # Loop over IC, KD, KH, KW
    for kd in tl.static_range(0, KD):
        id_num = od + PD - kd
        id_val = id_num // SD
        id_valid = (id_num >= 0) & ((id_num % SD) == 0) & (id_val < ID) & (id_val >= 0)
        for kh in tl.static_range(0, KH):
            ih_num = oh + PH - kh
            ih_val = ih_num // SH
            ih_valid = (ih_num >= 0) & ((ih_num % SH) == 0) & (ih_val < IH) & (ih_val >= 0)
            for kw in tl.static_range(0, KW):
                iw_num = ow + PW - kw
                iw_val = iw_num // SW
                iw_valid = (iw_num >= 0) & ((iw_num % SW) == 0) & (iw_val < IW) & (iw_val >= 0)

                spatial_valid = id_valid & ih_valid & iw_valid & sp_mask  # [BLOCK_SP]
                # Clamp indices to be safe under masked load
                id_safe = tl.where(id_valid, id_val, 0)
                ih_safe = tl.where(ih_valid, ih_val, 0)
                iw_safe = tl.where(iw_valid, iw_val, 0)

                for ic in tl.static_range(0, IC):
                    x_offset = ((pid_n * IC + ic) * ID + id_safe) * IH * IW + ih_safe * IW + iw_safe
                    x_val = tl.load(x_ptr + x_offset, mask=spatial_valid, other=0.0)  # [BLOCK_SP]

                    w_offset = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_offset, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    acc += x_val[:, None] * w_val[None, :]

    # Write output: out[n, oc, od, oh, ow]
    # out_offset[sp, oc] = ((n * OC + oc) * OD * OH * OW) + sp_offs
    out_off = (pid_n * OC + oc_offs)[None, :] * spatial + sp_offs[:, None]
    out_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


def conv_transpose3d_triton(x, weight, bias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    SD = SH = SW = stride
    PD = PH = PW = padding

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

    BLOCK_OC = 32
    BLOCK_SP = 32
    spatial = OD * OH * OW
    grid = (N, (OC + BLOCK_OC - 1) // BLOCK_OC, (spatial + BLOCK_SP - 1) // BLOCK_SP)

    conv_transpose3d_gemm_kernel[grid](
        x, weight, bias, out,
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
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
    N, C, SPATIAL,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    scale = tl.load(scale_ptr + c)
    shift = tl.load(shift_ptr + c)

    base = (n * C + c) * SPATIAL

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for off in range(0, SPATIAL, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < SPATIAL
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        y = x * scale + shift
        acc += tl.where(mask, y, 0.0)

    total = tl.sum(acc, axis=0)
    mean = total / SPATIAL

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

        y = self.batch_norm(y)

        N, C, D, H, W = y.shape
        scale = torch.ones(C, device=y.device, dtype=y.dtype)
        shift = torch.zeros(C, device=y.device, dtype=y.dtype)
        y = bn_meansub_triton(y.contiguous(), scale, shift)
        return y