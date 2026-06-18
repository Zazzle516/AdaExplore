import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _conv_transpose3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    BLOCK_SPATIAL: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, oc_tile, spatial_tile)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    OHW = OH * OW
    ODHW = OD * OHW

    sp_offs = pid_sp * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    sp_mask = sp_offs < ODHW

    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Accumulator [BLOCK_OC, BLOCK_SPATIAL]
    acc = tl.zeros((BLOCK_OC, BLOCK_SPATIAL), dtype=tl.float32)

    # For each output (od, oh, ow), we want to compute sum over (ic, kd, kh, kw):
    # in_d = (od + PD - kd) / SD  if divisible and in [0, ID)
    # similarly for h, w
    # output[n, oc, od, oh, ow] += input[n, ic, in_d, in_h, in_w] * weight[ic, oc, kd, kh, kw]

    d_plus = od + PD  # [BLOCK_SPATIAL]
    h_plus = oh + PH
    w_plus = ow + PW

    for kd in range(KD):
        rd = d_plus - kd
        in_d = rd // SD
        valid_d = (rd - in_d * SD == 0) & (in_d >= 0) & (in_d < ID)
        for kh in range(KH):
            rh = h_plus - kh
            in_h = rh // SH
            valid_h = (rh - in_h * SH == 0) & (in_h >= 0) & (in_h < IH)
            for kw in range(KW):
                rw = w_plus - kw
                in_w = rw // SW
                valid_w = (rw - in_w * SW == 0) & (in_w >= 0) & (in_w < IW)
                valid = valid_d & valid_h & valid_w & sp_mask

                # input index for [BLOCK_SPATIAL]
                in_idx_spatial = in_d * (IH * IW) + in_h * IW + in_w  # [BLOCK_SPATIAL]

                for ic in range(IC):
                    # Load input [BLOCK_SPATIAL]
                    x_ptrs = x_ptr + pid_n * (IC * ID * IH * IW) + ic * (ID * IH * IW) + in_idx_spatial
                    x_vals = tl.load(x_ptrs, mask=valid, other=0.0)  # [BLOCK_SPATIAL]

                    # Load weight [BLOCK_OC]
                    # weight shape: (IC, OC, KD, KH, KW)
                    w_ptrs = w_ptr + ic * (OC * KD * KH * KW) + oc_offs * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w_vals = tl.load(w_ptrs, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    acc += w_vals[:, None] * x_vals[None, :]

    # Add bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += bias[:, None]

    # Store
    out_ptrs = out_ptr + pid_n * (OC * ODHW) + oc_offs[:, None] * ODHW + sp_offs[None, :]
    store_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptrs, acc, mask=store_mask)


@triton.jit
def _fused_softmax_kernel(
    x_ptr, scale_ptr, out_ptr,
    S, C,
    clamp_min, clamp_max,
    BLOCK: tl.constexpr,
):
    bc = tl.program_id(0)
    c = bc % C
    row_start = bc * S

    max_val = -float('inf')
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=-float('inf'))
        v = tl.minimum(tl.maximum(v, clamp_min), clamp_max)
        v = tl.where(mask, v, -float('inf'))
        block_max = tl.max(v, axis=0)
        max_val = tl.maximum(max_val, block_max)

    sum_val = 0.0
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        v = tl.minimum(tl.maximum(v, clamp_min), clamp_max)
        e = tl.exp(v - max_val)
        e = tl.where(mask, e, 0.0)
        sum_val += tl.sum(e, axis=0)

    inv_sum = 1.0 / sum_val
    s = tl.load(scale_ptr + c)

    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        v = tl.minimum(tl.maximum(v, clamp_min), clamp_max)
        e = tl.exp(v - max_val) * inv_sum * s
        tl.store(out_ptr + row_start + idx, e, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super().__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        # Keep conv_transpose to extract weights/bias
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding, padding) if isinstance(padding, int) else padding
        self.output_padding = (output_padding, output_padding, output_padding) if isinstance(output_padding, int) else output_padding

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.contiguous()

        N, IC, ID, IH, IW = x.shape
        KD, KH, KW = self.kernel_size
        SD, SH, SW = self.stride
        PD, PH, PW = self.padding
        OPD, OPH, OPW = self.output_padding
        OC = self.out_channels

        OD = (ID - 1) * SD - 2 * PD + KD + OPD
        OH = (IH - 1) * SH - 2 * PH + KH + OPH
        OW = (IW - 1) * SW - 2 * PW + KW + OPW

        # Allocate output
        out_ct = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        weight = self.conv_transpose.weight  # (IC, OC, KD, KH, KW)
        bias = self.conv_transpose.bias  # (OC,)

        # Determine block sizes
        BLOCK_OC = 32
        BLOCK_SPATIAL = 64

        ODHW = OD * OH * OW
        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(ODHW, BLOCK_SPATIAL))

        _conv_transpose3d_kernel[grid](
            x, weight, bias, out_ct,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            BLOCK_SPATIAL=BLOCK_SPATIAL,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
            num_stages=2,
        )

        # Now fused clamp + softmax + scale
        S = OD * OH * OW
        x_flat = out_ct.view(N * OC, S)
        out = torch.empty_like(x_flat)
        scale_flat = self.scale.view(-1).contiguous()

        if S <= 1024:
            BLOCK = triton.next_power_of_2(S)
            num_warps_sm = 4 if BLOCK <= 256 else 8
        else:
            BLOCK = 1024
            num_warps_sm = 8

        grid_sm = (N * OC,)
        _fused_softmax_kernel[grid_sm](
            x_flat, scale_flat, out,
            S, OC,
            self.clamp_min, self.clamp_max,
            BLOCK=BLOCK,
            num_warps=num_warps_sm,
        )

        return out.view(N, OC, OD, OH, OW)