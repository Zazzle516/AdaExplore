import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _convt_pool_fused_kernel(
    x_ptr,           # input after avg_pool view: actually we pass raw x and do pool inline
    w_ptr,           # weight: (IC, OC, KD, KH, KW)
    b_ptr,           # bias: (OC,)
    out_ptr,         # output: (B, OC, OD, OH, OW)
    # input dims (raw, before pool)
    B, IC, ID_RAW, IH_RAW, IW_RAW,
    # pooled input dims
    ID, IH, IW,
    # output dims
    OC, OD, OH, OW,
    # constexpr
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    # grid: (B, ceil(OC/BLOCK_OC), ceil(OD*OH*OW/BLOCK_SP))
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (OD * OH * OW)

    # decode spatial index
    ow = sp_offs % OW
    tmp = sp_offs // OW
    oh = tmp % OH
    od = tmp // OH

    # accumulator
    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # for each output (od, oh, ow), iterate over kernel positions and find input pos
    # transposed conv relation: out[od] += sum over kd of in[id] * w[ic,oc,kd,...]
    #   where od = id * STRIDE - PAD + kd  =>  id = (od + PAD - kd) / STRIDE
    # only when (od + PAD - kd) divisible by STRIDE and in valid range
    for kd in tl.static_range(0, KD):
        id_num = od + PAD - kd
        id_div = id_num // STRIDE
        id_valid = ((id_num - id_div * STRIDE) == 0) & (id_div >= 0) & (id_div < ID)
        for kh in tl.static_range(0, KH):
            ih_num = oh + PAD - kh
            ih_div = ih_num // STRIDE
            ih_valid = ((ih_num - ih_div * STRIDE) == 0) & (ih_div >= 0) & (ih_div < IH)
            for kw in tl.static_range(0, KW):
                iw_num = ow + PAD - kw
                iw_div = iw_num // STRIDE
                iw_valid = ((iw_num - iw_div * STRIDE) == 0) & (iw_div >= 0) & (iw_div < IW)
                spatial_valid = id_valid & ih_valid & iw_valid & sp_mask  # [BLOCK_SP]

                # for each ic, gather input value (with avg pool inline) and multiply by weight
                # input value comes from avg_pool of raw x
                # pooled[id_div, ih_div, iw_div] = mean of raw[2*id_div+a, 2*ih_div+b, 2*iw_div+c]
                # We'll compute the sum of raw values then divide by 8 once.
                # base raw indices
                rd = id_div * 2
                rh = ih_div * 2
                rw = iw_div * 2

                for ic in range(0, IC):
                    # load 8 raw values and average
                    base = ((pid_b * IC + ic) * ID_RAW) * IH_RAW * IW_RAW
                    # raw layout: [B, IC, ID_RAW, IH_RAW, IW_RAW]
                    # offset: base + rd*IH_RAW*IW_RAW + rh*IW_RAW + rw
                    sum_raw = tl.zeros((BLOCK_SP,), dtype=tl.float32)
                    for da in tl.static_range(0, 2):
                        for db in tl.static_range(0, 2):
                            for dc in tl.static_range(0, 2):
                                rd_i = rd + da
                                rh_i = rh + db
                                rw_i = rw + dc
                                in_off = base + rd_i * IH_RAW * IW_RAW + rh_i * IW_RAW + rw_i
                                v = tl.load(x_ptr + in_off, mask=spatial_valid, other=0.0)
                                sum_raw = sum_raw + v
                    in_val = sum_raw * 0.125  # [BLOCK_SP]

                    # weight: w[ic, oc, kd, kh, kw]
                    w_off = (((ic * OC + oc_offs) * KD + kd) * KH + kh) * KW + kw
                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    # outer product accumulate
                    acc += in_val[:, None] * w_val[None, :]

    # add bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[None, :]

    # apply spatial valid mask (zero out invalid sp)
    acc = tl.where(sp_mask[:, None] & oc_mask[None, :], acc, 0.0)

    # store: out[b, oc, od, oh, ow]
    out_base = (pid_b * OC) * OD * OH * OW
    out_off = out_base + oc_offs[None, :] * (OD * OH * OW) + sp_offs[:, None]
    out_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


@triton.jit
def _clamp_softmax_scale_kernel(
    x_ptr, scale_ptr, out_ptr,
    S,
    CLAMP_MIN: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    C = tl.num_programs(1)
    row_start = (b * C + c) * S
    scale_val = tl.load(scale_ptr + c)

    max_val = -float('inf')
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=-float('inf'))
        v = tl.minimum(tl.maximum(v, CLAMP_MIN), CLAMP_MAX)
        v = tl.where(mask, v, -float('inf'))
        m = tl.max(v, axis=0)
        max_val = tl.maximum(max_val, m)

    sum_val = 0.0
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        v = tl.minimum(tl.maximum(v, CLAMP_MIN), CLAMP_MAX)
        e = tl.exp(v - max_val)
        e = tl.where(mask, e, 0.0)
        sum_val += tl.sum(e, axis=0)

    inv_sum = 1.0 / sum_val

    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        v = tl.minimum(tl.maximum(v, CLAMP_MIN), CLAMP_MAX)
        e = tl.exp(v - max_val) * inv_sum * scale_val
        tl.store(out_ptr + row_start + idx, e, mask=mask)


def fused_clamp_softmax_scale(x, scale, clamp_min, clamp_max):
    B, C, D, H, W = x.shape
    S = D * H * W
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    scale_flat = scale.contiguous().view(-1)

    BLOCK = 1024
    grid = (B, C)
    _clamp_softmax_scale_kernel[grid](
        x_c, scale_flat, out,
        S,
        float(clamp_min), float(clamp_max),
        BLOCK=BLOCK,
        num_warps=8,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        # Use torch's optimized avg_pool + conv_transpose (cuDNN) — beat baseline by fusing tail
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        x = fused_clamp_softmax_scale(x, self.scale, self.clamp_min, self.clamp_max)
        return x