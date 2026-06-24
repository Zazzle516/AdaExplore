import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SPATIAL': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SPATIAL': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SPATIAL': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SPATIAL': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SPATIAL': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SPATIAL': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_SPATIAL': 64}, num_warps=4, num_stages=2),
    ],
    key=['N', 'IC', 'OC', 'OD', 'OH', 'OW', 'KD', 'KH', 'KW'],
)
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
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    ic_offs = tl.arange(0, BLOCK_IC)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (OD * OH * OW)
    ic_mask = ic_offs < IC

    od = sp_offs // (OH * OW)
    rem = sp_offs % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    bias_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = tl.zeros((BLOCK_OC, BLOCK_SPATIAL), dtype=tl.float32)

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

                # Load x tile [BLOCK_IC, BLOCK_SPATIAL]
                # x[n, ic, id_idx, ih_idx, iw_idx]
                x_spat_off = id_idx * (IH * IW) + ih_idx * IW + iw_idx  # [BLOCK_SPATIAL]
                x_base = pid_n * IC * ID * IH * IW
                x_off = x_base + ic_offs[:, None] * (ID * IH * IW) + x_spat_off[None, :]
                x_load_mask = ic_mask[:, None] & spatial_valid[None, :]
                x_tile = tl.load(x_ptr + x_off, mask=x_load_mask, other=0.0)

                # Load w tile [BLOCK_OC, BLOCK_IC]
                # w[ic, oc, kd, kh, kw]
                w_off = ic_offs[None, :] * (OC * KD * KH * KW) + oc_offs[:, None] * (KD * KH * KW) + kd * KH * KW + kh * KW + kw
                w_load_mask = oc_mask[:, None] & ic_mask[None, :]
                w_tile = tl.load(w_ptr + w_off, mask=w_load_mask, other=0.0)

                acc += tl.dot(w_tile, x_tile, allow_tf32=False)

    acc += bias_vals[:, None]

    out_off = ((pid_n * OC + oc_offs[:, None]) * OD * OH * OW) + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


@triton.jit
def bn_meansub_fused_kernel(
    x_ptr, out_ptr,
    SPATIAL,
    BLOCK: tl.constexpr,
):
    # one program per (n, oc): two-pass sum then subtract
    pid = tl.program_id(0)
    base = pid * SPATIAL

    sum_val = tl.zeros((), dtype=tl.float32)
    for off in range(0, SPATIAL, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < SPATIAL
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)

    mean = sum_val / SPATIAL

    for off in range(0, SPATIAL, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < SPATIAL
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        tl.store(out_ptr + base + idx, x - mean, mask=mask)


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

        # Pad BLOCK_IC to nearest power of 2 >= IC, min 16
        BLOCK_IC = 16
        while BLOCK_IC < IC:
            BLOCK_IC *= 2

        grid = lambda META: (N, triton.cdiv(OC, META['BLOCK_OC']), triton.cdiv(SPATIAL, META['BLOCK_SPATIAL']))
        conv_transpose3d_kernel[grid](
            x, weight, bias, conv_out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            BLOCK_IC=BLOCK_IC,
        )

        # BN: running on training=True. Use functional with training=True to use batch stats.
        bn_out = self.batch_norm(conv_out)

        # Fused sum + mean-subtract in a single kernel (one program per (N,OC))
        out = torch.empty_like(bn_out)

        BLOCK_RED = 2048
        bn_meansub_fused_kernel[(N * OC,)](
            bn_out, out,
            SPATIAL,
            BLOCK=BLOCK_RED,
            num_warps=8,
            num_stages=2,
        )

        return out