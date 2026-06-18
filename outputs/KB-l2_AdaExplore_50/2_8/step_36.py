import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Conv3d kernel: implicit GEMM-style.
# Output shape: [N, OC, OD, OH, OW] where OD=ID-2, OH=IH-2, OW=IW-2 (kernel 3x3x3, no padding).
# We tile over (N, OC, OD*OH*OW spatial flattened).
# Each program computes a tile of BLOCK_M output spatial positions for BLOCK_N output channels for one batch.
# Uses tl.dot with K-dim = IC * KD * KH * KW (= 8*27 = 216 for this problem).

@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    inv_div,
    OHW, ODHW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
):
    pid_n = tl.program_id(0)             # batch
    pid_oc = tl.program_id(1)            # OC tile
    pid_sp = tl.program_id(2)            # spatial tile

    sp_offs = pid_sp * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    sp_mask = sp_offs < ODHW

    # Decompose sp_offs -> (od, oh, ow)
    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    oc_offs = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    oc_mask = oc_offs < OC

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Loop over IC, KD, KH, KW
    KDHW = KD * KH * KW
    for ic in range(0, IC_C):
        for kd in range(0, KD):
            for kh in range(0, KH):
                for kw in range(0, KW):
                    # input position
                    id_ = od + kd     # [BLOCK_M]
                    ih = oh + kh
                    iw = ow + kw
                    x_off = (((pid_n * IC + ic) * ID + id_) * IH + ih) * IW + iw
                    x_v = tl.load(x_ptr + x_off, mask=sp_mask, other=0.0)  # [BLOCK_M]

                    w_off = ((oc_offs * IC + ic) * KD + kd) * KH * KW + kh * KW + kw
                    w_v = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_N]

                    acc += x_v[:, None] * w_v[None, :]

    # Add bias
    b_v = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_N]
    acc = (acc + b_v[None, :]) * inv_div

    # Store: out[N, OC, OD, OH, OW]
    out_off = ((pid_n * OC + oc_offs[None, :]) * ODHW) + sp_offs[:, None]
    mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask)


# Fused kernel: maxpool 2x2x2 -> global avg pool -> +bias -> sum over channel
# Input: [N, OC, OD, OH, OW]
# Output: [N] scalar per batch (after sum over channel dim)
@triton.jit
def pool_reduce_kernel(
    inp_ptr, bias_ptr, out_ptr,
    N, OC, OD, OH, OW,
    PD, PH, PW,
    BLOCK_OC: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    n = tl.program_id(0)
    oc_block = tl.program_id(1)

    oc_offs = oc_block * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    oc_mask = oc_offs < OC

    PHW = PH * PW
    PDHW = PD * PH * PW
    OHW = OH * OW
    ODHW = OD * OH * OW

    # Accumulator for max-pool sum over all pooled positions, per channel
    acc = tl.zeros([BLOCK_OC], dtype=tl.float32)

    # Iterate over pooled positions in tiles
    for p_start in range(0, PDHW, BLOCK_P):
        p_offs = p_start + tl.arange(0, BLOCK_P)  # [BLOCK_P]
        p_mask = p_offs < PDHW

        pd = p_offs // PHW
        rem = p_offs % PHW
        ph = rem // PW
        pw = rem % PW

        # max over 2x2x2 window
        max_val = tl.full([BLOCK_OC, BLOCK_P], -1e38, dtype=tl.float32)
        for dd in range(0, 2):
            for dh in range(0, 2):
                for dw in range(0, 2):
                    od = pd * 2 + dd
                    oh = ph * 2 + dh
                    ow = pw * 2 + dw
                    # input offset: [N, OC, OD, OH, OW]
                    in_off = ((n * OC + oc_offs[:, None]) * ODHW) + (od[None, :] * OHW + oh[None, :] * OW + ow[None, :])
                    mask = oc_mask[:, None] & p_mask[None, :]
                    v = tl.load(inp_ptr + in_off, mask=mask, other=-1e38)
                    max_val = tl.maximum(max_val, v)

        # mask out invalid p positions
        max_val = tl.where(p_mask[None, :], max_val, 0.0)
        # sum over BLOCK_P
        acc += tl.sum(max_val, axis=1)

    # global avg = acc / PDHW
    avg = acc / PDHW
    # add bias (per-channel) - bias shape (OC,1,1,1) -> per channel
    bias_v = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    avg = avg + bias_v
    # sum across OC
    avg = tl.where(oc_mask, avg, 0.0)
    partial = tl.sum(avg, axis=0)
    tl.atomic_add(out_ptr + n, partial)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kd = self.kh = self.kw = kernel_size
        else:
            self.kd, self.kh, self.kw = kernel_size
        if isinstance(pool_size, int):
            self.pd = self.ph = self.pw = pool_size
        else:
            self.pd, self.ph, self.pw = pool_size

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD, KH, KW = self.kd, self.kh, self.kw
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1

        weight = self.conv.weight.contiguous()
        cbias = self.conv.bias.contiguous()

        conv_out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

        BLOCK_M = 64
        BLOCK_N = 16  # OC=16 fits exactly
        ODHW = OD * OH * OW
        OHW = OH * OW

        grid_conv = (N, triton.cdiv(OC, BLOCK_N), triton.cdiv(ODHW, BLOCK_M))
        conv3d_kernel[grid_conv](
            x, weight, cbias, conv_out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            1.0 / self.divisor,
            OHW, ODHW,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            KD=KD, KH=KH, KW=KW,
            IC_C=IC,
            num_warps=4, num_stages=2,
        )

        # Pool dims
        PD = OD // self.pd
        PH = OH // self.ph
        PW = OW // self.pw

        # Output: sum over channel of (avgpool + bias). Result shape: per-batch scalar (after sum_dim=1)
        # Original output shape after sum: [N, 1, 1, 1] (since global avg gave 1,1,1 and we sum over dim=1)
        out = torch.zeros((N,), device=x.device, dtype=torch.float32)

        bias_flat = self.bias.view(-1).contiguous()

        BLOCK_OC = 16
        BLOCK_P = 64
        grid_pool = (N, triton.cdiv(OC, BLOCK_OC))
        pool_reduce_kernel[grid_pool](
            conv_out, bias_flat, out,
            N, OC, OD, OH, OW,
            PD, PH, PW,
            BLOCK_OC=BLOCK_OC, BLOCK_P=BLOCK_P,
            num_warps=4, num_stages=2,
        )

        return out.view(N, 1, 1, 1)