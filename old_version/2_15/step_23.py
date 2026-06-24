import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Forward conv (gather-style) where weight has been pre-transformed to
# weight_flipped[OC, IC, KD, KH, KW] = weight_orig[IC, OC, KD-1-kd, KH-1-kh, KW-1-kw]
# Then ConvTranspose3d output = "regular conv" with this flipped weight applied
# at dilated/padded input positions.
# But it's simpler to keep gather formulation:
#   out[n, oc, od, oh, ow] = sum_{ic, kd, kh, kw}
#       w[ic, oc, kd, kh, kw] * x[n, ic, id, ih, iw]
#   where id = (od + PD - kd) / SD if divisible.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_SP': 128}, num_warps=4, num_stages=2),
    ],
    key=['IC', 'OC', 'OD', 'OH', 'OW', 'KD', 'KH', 'KW'],
)
@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    OHW = OH * OW
    SPATIAL = OD * OHW

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < SPATIAL

    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    bias_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    IHW = IH * IW

    for kd in tl.static_range(KD):
        id_num = od + PD - kd
        id_idx = id_num // SD
        id_valid = (id_num >= 0) & ((id_num - id_idx * SD) == 0) & (id_idx >= 0) & (id_idx < ID)
        for kh in tl.static_range(KH):
            ih_num = oh + PH - kh
            ih_idx = ih_num // SH
            ih_valid = (ih_num >= 0) & ((ih_num - ih_idx * SH) == 0) & (ih_idx >= 0) & (ih_idx < IH)
            for kw in tl.static_range(KW):
                iw_num = ow + PW - kw
                iw_idx = iw_num // SW
                iw_valid = (iw_num >= 0) & ((iw_num - iw_idx * SW) == 0) & (iw_idx >= 0) & (iw_idx < IW)

                spatial_valid = id_valid & ih_valid & iw_valid & sp_mask

                # Per-spatial input base offset (within an N,IC slice it's just spatial; we add ic*ID*IH*IW)
                in_sp_off = id_idx * IHW + ih_idx * IW + iw_idx  # [BLOCK_SP]

                for ic in range(IC):
                    x_off = (pid_n * IC + ic) * ID * IHW + in_sp_off
                    x_val = tl.load(x_ptr + x_off, mask=spatial_valid, other=0.0)

                    # w shape: [IC, OC, KD, KH, KW]
                    w_off = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)

                    acc += w_val[:, None] * x_val[None, :]

    acc = acc + bias_vals[:, None]

    out_off = ((pid_n * OC + oc_offs[:, None]) * SPATIAL) + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


# Fused kernel:
#   - Apply BN affine: y = x * scale + shift, where (scale, shift) come from
#     batch statistics computed in PyTorch's BN training mode (we replicate it here).
#   - Subtract per-(n, oc) spatial mean.
# To match training-mode BN, we compute per-channel batch mean/var across (N, D, H, W),
# then apply normalization, then subtract per-(N, OC) spatial mean.
#
# Since for our specific reduction (subtracting per-(n,oc) mean), the BN affine's
# additive `shift` term cancels out within the per-(n,oc) mean subtraction,
# but the multiplicative `scale` does not. Still, do it explicitly to match BN.

@triton.jit
def fused_bn_meansub_kernel(
    x_ptr, scale_ptr, shift_ptr, out_ptr,
    N, OC, SPATIAL,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid % OC

    scale = tl.load(scale_ptr + oc)
    shift = tl.load(shift_ptr + oc)

    base = (n * OC + oc) * SPATIAL

    # Pass 1: sum of normalized values
    sum_val = tl.zeros((), dtype=tl.float32)
    for off in range(0, SPATIAL, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < SPATIAL
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        normed = x * scale + shift
        sum_val += tl.sum(tl.where(mask, normed, 0.0))

    mean = sum_val / SPATIAL

    # Pass 2: write
    for off in range(0, SPATIAL, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < SPATIAL
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        normed = x * scale + shift
        tl.store(out_ptr + base + idx, normed - mean, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
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

        SPATIAL = OD * OH * OW

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_OC']),
            triton.cdiv(SPATIAL, meta['BLOCK_SP']),
        )
        conv_transpose3d_kernel[grid](
            x, weight, bias, conv_out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
        )

        # Apply BatchNorm (training-mode) using PyTorch (it updates running stats and uses batch stats)
        bn_out = self.batch_norm(conv_out)

        # Fused mean subtraction along (D,H,W). Note BN already applied; we just subtract spatial mean.
        out = torch.empty_like(bn_out)

        # Use a simple kernel: one program per (n, oc), single-pass would need full row in regs.
        # Reuse fused_bn_meansub but with scale=1, shift=0 (i.e., just spatial mean subtract).
        scale = torch.ones(OC, device=x.device, dtype=x.dtype)
        shift = torch.zeros(OC, device=x.device, dtype=x.dtype)

        BLOCK = 1024
        fused_bn_meansub_kernel[(N * OC,)](
            bn_out, scale, shift, out,
            N, OC, SPATIAL,
            BLOCK=BLOCK,
            num_warps=8,
            num_stages=2,
        )

        return out