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
    BLOCK_SPATIAL: tl.constexpr,
):
    # program ids: (n, oc_block, spatial_block)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (OD * OH * OW)

    od = sp_offs // (OH * OW)
    rem = sp_offs % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    # Initialize accumulator with bias
    bias_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = tl.zeros((BLOCK_OC, BLOCK_SPATIAL), dtype=tl.float32) + bias_vals[:, None]

    # For each output position (od, oh, ow), iterate over kernel positions
    # input position: id*SD - PD + kd = od  =>  id = (od + PD - kd) / SD
    # must be divisible by SD and within [0, ID)
    for kd in range(KD):
        id_num = od + PD - kd
        id_idx = id_num // SD
        id_valid = (id_num >= 0) & (id_num % SD == 0) & (id_idx >= 0) & (id_idx < ID)
        for kh in range(KH):
            ih_num = oh + PH - kh
            ih_idx = ih_num // SH
            ih_valid = (ih_num >= 0) & (ih_num % SH == 0) & (ih_idx >= 0) & (ih_idx < IH)
            for kw in range(KW):
                iw_num = ow + PW - kw
                iw_idx = iw_num // SW
                iw_valid = (iw_num >= 0) & (iw_num % SW == 0) & (iw_idx >= 0) & (iw_idx < IW)

                spatial_valid = id_valid & ih_valid & iw_valid & sp_mask

                for ic in range(IC):
                    # Load x[n, ic, id, ih, iw]
                    x_off = ((pid_n * IC + ic) * ID + id_idx) * IH * IW + ih_idx * IW + iw_idx
                    x_val = tl.load(x_ptr + x_off, mask=spatial_valid, other=0.0)

                    # Load w[ic, oc, kd, kh, kw] for all oc in block
                    w_off = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)

                    acc += w_val[:, None] * x_val[None, :]

    # Store: out[n, oc, od, oh, ow]
    out_off = ((pid_n * OC + oc_offs[:, None]) * OD * OH * OW) + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


@triton.jit
def bn_meansub_kernel(
    x_ptr, scale_ptr, shift_ptr, out_ptr,
    N, OC, SPATIAL,
    BLOCK: tl.constexpr,
):
    # one program per (n, oc), does normalization + mean subtraction
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid % OC

    scale = tl.load(scale_ptr + oc)
    shift = tl.load(shift_ptr + oc)

    base = (n * OC + oc) * SPATIAL

    # First pass: compute sum of normalized values
    sum_val = tl.zeros((), dtype=tl.float32)
    for off in range(0, SPATIAL, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < SPATIAL
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        normed = x * scale + shift
        sum_val += tl.sum(tl.where(mask, normed, 0.0))

    mean = sum_val / SPATIAL

    # Second pass: subtract mean and write
    for off in range(0, SPATIAL, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < SPATIAL
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        normed = x * scale + shift
        out = normed - mean
        tl.store(out_ptr + base + idx, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Match ConvTranspose3d default param init
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous().cuda()
        if self.conv_transpose.bias is not None:
            bias = self.conv_transpose.bias.contiguous().cuda()
        else:
            bias = torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)

        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        SD = SH = SW = self.stride
        PD = PH = PW = self.padding

        OD = (ID - 1) * SD - 2 * PD + KD
        OH = (IH - 1) * SH - 2 * PH + KH
        OW = (IW - 1) * SW - 2 * PW + KW

        conv_out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_SPATIAL = 64
        SPATIAL = OD * OH * OW

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(SPATIAL, BLOCK_SPATIAL))
        conv_transpose3d_kernel[grid](
            x, weight, bias, conv_out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            BLOCK_OC=BLOCK_OC,
            BLOCK_SPATIAL=BLOCK_SPATIAL,
            num_warps=4,
            num_stages=2,
        )

        # BN at training=True uses batch statistics. But we're not doing training here;
        # to match reference, we need to apply BN as it would in training mode.
        # The reference calls self.batch_norm(x) which in training mode computes batch stats.
        # We'll just call F.batch_norm with training=True semantics via the module.
        bn_out = self.batch_norm(conv_out)

        # Mean subtract along (2,3,4)
        out = torch.empty_like(bn_out)
        # Use a simpler approach: just compute mean and subtract
        mean = bn_out.mean(dim=(2, 3, 4), keepdim=True)
        out = bn_out - mean

        return out