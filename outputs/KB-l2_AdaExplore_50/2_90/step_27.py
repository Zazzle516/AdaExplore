import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv3d_fused_kernel(
    x_ptr,        # [N, IC, ID, IH, IW]
    w_ptr,        # [OC, IC*KD*KH*KW]
    b_ptr,        # [OC]
    sum_ptr,      # [OC]
    out_ptr,      # [N, OC, OD, OH, OW]
    N, IC, ID, IH, IW,
    OD, OH, OW,
    OC: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    K: tl.constexpr,         # IC*KD*KH*KW
    BLOCK_N: tl.constexpr,   # spatial tile size
):
    pid_n = tl.program_id(0)        # batch index
    pid_s = tl.program_id(1)        # spatial tile

    offs_oc = tl.arange(0, OC)              # [OC]
    offs_s = pid_s * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    OHW = OH * OW
    ODHW = OD * OHW

    s_mask = offs_s < ODHW

    # decompose spatial index into (od, oh, ow)
    od = offs_s // OHW
    rem = offs_s % OHW
    oh = rem // OW
    ow = rem % OW

    # accumulator [OC, BLOCK_N]
    acc = tl.zeros((OC, BLOCK_N), dtype=tl.float32)

    # Input base for this batch
    x_batch_off = pid_n * IC * ID * IH * IW

    # Loop over K = IC * KD * KH * KW
    for k in tl.static_range(0, K):
        kw = k % KW
        khw = k // KW
        kh = khw % KH
        kdh = khw // KH
        kd = kdh % KD
        ic = kdh // KD

        # input spatial coords
        id_ = od + kd
        ih_ = oh + kh
        iw_ = ow + kw

        in_off = x_batch_off + ic * (ID * IH * IW) + id_ * (IH * IW) + ih_ * IW + iw_
        x_val = tl.load(x_ptr + in_off, mask=s_mask, other=0.0)  # [BLOCK_N]

        # weight column [OC]
        w_col = tl.load(w_ptr + offs_oc * K + k)  # [OC]

        acc += w_col[:, None] * x_val[None, :]

    # bias
    bias = tl.load(b_ptr + offs_oc)  # [OC]
    acc = acc + bias[:, None]

    # leaky relu
    acc = tl.where(acc > 0, acc, acc * 0.2)

    # sum tensor add (per OC)
    s_val = tl.load(sum_ptr + offs_oc)  # [OC]
    acc = acc + s_val[:, None]

    # clamp
    acc = tl.minimum(tl.maximum(acc, -1.0), 1.0)

    # GELU exact
    inv_sqrt2 = 0.7071067811865475
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # store: out[n, oc, s]
    out_base = pid_n * OC * ODHW + offs_oc[:, None] * ODHW + offs_s[None, :]
    out_mask = s_mask[None, :]
    tl.store(out_ptr + out_base, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        OC = self.out_channels
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1
        K = IC * KD * KH * KW

        # pack weight to [OC, K]
        w = self.conv.weight.contiguous().view(OC, K).contiguous()
        b = self.conv.bias.contiguous()
        s = self.sum_tensor.contiguous().view(-1).contiguous()

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        ODHW = OD * OH * OW
        BLOCK_N = 128
        grid = (N, triton.cdiv(ODHW, BLOCK_N))

        conv3d_fused_kernel[grid](
            x, w, b, s, out,
            N, IC, ID, IH, IW,
            OD, OH, OW,
            OC=OC,
            KD=KD, KH=KH, KW=KW,
            K=K,
            BLOCK_N=BLOCK_N,
            num_warps=8,
            num_stages=2,
        )
        return out