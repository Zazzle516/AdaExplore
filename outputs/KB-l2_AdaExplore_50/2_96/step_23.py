import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def convT3d_scatter_kernel(
    x_ptr,           # (N, IC, ID, IH, IW)
    w_ptr,           # (IC, OC, KD, KH, KW)
    b_ptr,           # (OC,)
    y_ptr,           # (N, OC, OD, OH, OW)
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # grid: (N * ID * IH * IW, IC, ceil(OC / BLOCK_OC))? 
    # Simpler: one program per (N, IC, ID*IH*IW), iterate OC tile inside.
    pid_nic = tl.program_id(0)  # n*IC + ic
    pid_spatial = tl.program_id(1)  # id*IH*IW + ih*IW + iw
    pid_oc = tl.program_id(2)  # OC tile

    n = pid_nic // IC
    ic = pid_nic % IC

    iw = pid_spatial % IW
    tmp = pid_spatial // IW
    ih = tmp % IH
    id_ = tmp // IH

    # Load input value
    x_off = ((n * IC + ic) * ID + id_) * IH * IW + ih * IW + iw
    x_val = tl.load(x_ptr + x_off)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # weight base: ic * OC * KD * KH * KW
    # For each (kd, kh, kw) compute output position and scatter
    for kd in tl.static_range(0, KD):
        od = id_ * STRIDE - PAD + kd
        for kh in tl.static_range(0, KH):
            oh = ih * STRIDE - PAD + kh
            for kw in tl.static_range(0, KW):
                ow = iw * STRIDE - PAD + kw
                in_bounds = (od >= 0) & (od < OD) & (oh >= 0) & (oh < OH) & (ow >= 0) & (ow < OW)
                # load weight for all oc in tile
                w_off = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                contrib = x_val * w_val
                y_off = ((n * OC + oc_offs) * OD + od) * OH * OW + oh * OW + ow
                mask = oc_mask & in_bounds
                tl.atomic_add(y_ptr + y_off, contrib, mask=mask)


@triton.jit
def init_with_bias_kernel(
    y_ptr,
    b_ptr,
    N, OC, SPATIAL,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # over N*OC
    n = pid // OC
    oc = pid % OC
    b = tl.load(b_ptr + oc)
    base = (n * OC + oc) * SPATIAL
    for off in range(0, SPATIAL, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < SPATIAL
        tl.store(y_ptr + base + idx, b, mask=mask)


@triton.jit
def fused_maxpool_mean_clamp_kernel(
    x_ptr,
    out_ptr,
    N, C, D, H, W,
    Dp, Hp, Wp,
    scale,
    inv_count,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    nc_base = (n * C + c) * D * H * W
    pooled_total = Dp * Hp * Wp

    acc = 0.0
    for off in range(0, pooled_total, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < pooled_total
        pw = idx % Wp
        tmp = idx // Wp
        ph = tmp % Hp
        pd = tmp // Hp

        d0 = pd * 2
        h0 = ph * 2
        w0 = pw * 2

        b000 = nc_base + d0 * H * W + h0 * W + w0
        b001 = b000 + 1
        b010 = b000 + W
        b011 = b010 + 1
        b100 = b000 + H * W
        b101 = b100 + 1
        b110 = b100 + W
        b111 = b110 + 1

        v0 = tl.load(x_ptr + b000, mask=mask, other=-1e30)
        v1 = tl.load(x_ptr + b001, mask=mask, other=-1e30)
        v2 = tl.load(x_ptr + b010, mask=mask, other=-1e30)
        v3 = tl.load(x_ptr + b011, mask=mask, other=-1e30)
        v4 = tl.load(x_ptr + b100, mask=mask, other=-1e30)
        v5 = tl.load(x_ptr + b101, mask=mask, other=-1e30)
        v6 = tl.load(x_ptr + b110, mask=mask, other=-1e30)
        v7 = tl.load(x_ptr + b111, mask=mask, other=-1e30)

        m = tl.maximum(tl.maximum(tl.maximum(v0, v1), tl.maximum(v2, v3)),
                       tl.maximum(tl.maximum(v4, v5), tl.maximum(v6, v7)))
        m = tl.where(mask, m, 0.0)
        acc += tl.sum(m, axis=0)

    mean = acc * inv_count * scale
    mean = tl.minimum(tl.maximum(mean, 0.0), 1.0)
    tl.store(out_ptr + n * C + c, mean)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.clamp_min = 0
        self.clamp_max = 1
        self.maxpool_kernel_size = maxpool_kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        S = self.stride
        P = self.padding

        OD = (ID - 1) * S - 2 * P + KD
        OH = (IH - 1) * S - 2 * P + KH
        OW = (IW - 1) * S - 2 * P + KW

        weight = self.conv_transpose.weight  # (IC, OC, KD, KH, KW)
        bias = self.conv_transpose.bias       # (OC,)

        y = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        SPATIAL = OD * OH * OW
        grid_init = (N * OC,)
        init_with_bias_kernel[grid_init](
            y, bias, N, OC, SPATIAL,
            BLOCK=1024, num_warps=4,
        )

        BLOCK_OC = 16
        grid = (N * IC, ID * IH * IW, (OC + BLOCK_OC - 1) // BLOCK_OC)
        convT3d_scatter_kernel[grid](
            x, weight, bias, y,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            S, P,
            BLOCK_OC=BLOCK_OC,
            num_warps=2,
        )

        k = self.maxpool_kernel_size
        Dp, Hp, Wp = OD // k, OH // k, OW // k

        out = torch.empty((N, OC, 1, 1, 1), device=x.device, dtype=x.dtype)

        if k == 2:
            pooled_total = Dp * Hp * Wp
            if pooled_total <= 256:
                BLOCK = 256
            elif pooled_total <= 512:
                BLOCK = 512
            elif pooled_total <= 1024:
                BLOCK = 1024
            else:
                BLOCK = 2048
            inv_count = 1.0 / float(pooled_total)
            grid2 = (N * OC,)
            fused_maxpool_mean_clamp_kernel[grid2](
                y, out,
                N, OC, OD, OH, OW,
                Dp, Hp, Wp,
                float(self.scale),
                inv_count,
                BLOCK=BLOCK,
                num_warps=4,
            )
        else:
            y2 = self.maxpool(y * self.scale)
            mean = y2.mean(dim=[2, 3, 4], keepdim=True)
            out = torch.clamp(mean, 0.0, 1.0)

        return out