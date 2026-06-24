import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 8, 'BLOCK_SPATIAL': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_OC': 8, 'BLOCK_SPATIAL': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 8, 'BLOCK_SPATIAL': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_OC': 8, 'BLOCK_SPATIAL': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 8, 'BLOCK_SPATIAL': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_SPATIAL': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_SPATIAL': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_SPATIAL': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_SPATIAL': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_SPATIAL': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SPATIAL': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SPATIAL': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SPATIAL': 64}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'OC', 'OD', 'OH', 'OW'],
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
    BLOCK_SPATIAL: tl.constexpr,
):
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

    bias_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = tl.zeros((BLOCK_OC, BLOCK_SPATIAL), dtype=tl.float32) + bias_vals[:, None]

    # IC loop outermost to keep accumulator persistent
    for ic in range(IC):
        x_base_n_ic = (pid_n * IC + ic) * ID * IH * IW
        w_base_ic = ic * OC * KD * KH * KW
        for kd in tl.static_range(KD):
            id_num = od + PD - kd
            id_idx = id_num // SD
            id_valid = (id_num >= 0) & (id_num % SD == 0) & (id_idx >= 0) & (id_idx < ID)
            for kh in tl.static_range(KH):
                ih_num = oh + PH - kh
                ih_idx = ih_num // SH
                ih_valid = (ih_num >= 0) & (ih_num % SH == 0) & (ih_idx >= 0) & (ih_idx < IH)
                for kw in tl.static_range(KW):
                    iw_num = ow + PW - kw
                    iw_idx = iw_num // SW
                    iw_valid = (iw_num >= 0) & (iw_num % SW == 0) & (iw_idx >= 0) & (iw_idx < IW)

                    spatial_valid = id_valid & ih_valid & iw_valid & sp_mask

                    x_off = x_base_n_ic + id_idx * IH * IW + ih_idx * IW + iw_idx
                    x_val = tl.load(x_ptr + x_off, mask=spatial_valid, other=0.0)

                    w_off = w_base_ic + oc_offs * KD * KH * KW + kd * KH * KW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)

                    acc += w_val[:, None] * x_val[None, :]

    out_off = ((pid_n * OC + oc_offs[:, None]) * OD * OH * OW) + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


@triton.jit
def fused_bn_meansub_kernel(
    x_ptr, scale_ptr, shift_ptr, out_ptr,
    SPATIAL,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # n * OC + oc
    oc = pid % tl.num_programs(0)  # placeholder; replaced via parameter
    # We will instead pass OC as a param. Recompute:
    # (simpler: use pid // SPATIAL_PER_NC etc., but here we just need oc)


@triton.jit
def fused_bn_meansub_kernel2(
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
    inv_spatial = 1.0 / SPATIAL

    # First pass: compute sum (we just need mean of normed = scale*mean(x) + shift)
    sum_val = tl.zeros((), dtype=tl.float32)
    for off in range(0, SPATIAL, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < SPATIAL
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x)

    mean_x = sum_val * inv_spatial
    # mean of normed = scale * mean_x + shift
    # normed - mean_normed = scale * (x - mean_x)
    # So we don't even need shift in second pass

    for off in range(0, SPATIAL, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < SPATIAL
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        out = scale * (x - mean_x)
        tl.store(out_ptr + base + idx, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

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

        SPATIAL = OD * OH * OW

        grid = lambda META: (N, triton.cdiv(OC, META['BLOCK_OC']), triton.cdiv(SPATIAL, META['BLOCK_SPATIAL']))
        conv_transpose3d_kernel[grid](
            x, weight, bias, conv_out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
        )

        # Run BatchNorm3d (training=True uses batch stats); then fuse mean subtraction.
        # We need the same numerical result as: bn(x) - mean(bn(x), dim=(2,3,4)).
        # In training mode, BN computes per-channel batch stats and applies affine.
        # The fused kernel below normalizes via (x - mean_x) * scale, but BN already
        # subtracted the per-channel batch mean. So we just need: out = bn_out - per-NC-mean.
        # Implement that as a single fused pass.
        bn_out = self.batch_norm(conv_out)

        out = torch.empty_like(bn_out)
        # Fused per-(N,OC) mean subtraction
        # scale=1, shift=0 effectively (we just subtract mean)
        scale = torch.ones(OC, device=bn_out.device, dtype=bn_out.dtype)
        shift = torch.zeros(OC, device=bn_out.device, dtype=bn_out.dtype)

        BLOCK = 1024
        grid2 = (N * OC,)
        fused_bn_meansub_kernel2[grid2](
            bn_out, scale, shift, out,
            N, OC, SPATIAL,
            BLOCK=BLOCK,
            num_warps=4,
        )

        return out