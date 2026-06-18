import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def convt3d_scatter_kernel(
    x_ptr,        # (N, IC, ID, IH, IW)
    w_ptr,        # (IC, OC, KD, KH, KW)
    b_ptr,        # (OC,)
    out_ptr,      # (N, OC, OD, OH, OW)
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, ic, id, ih, iw)
    pid = tl.program_id(0)
    pid_oc = tl.program_id(1)

    iw = pid % IW
    tmp = pid // IW
    ih = tmp % IH
    tmp = tmp // IH
    id_ = tmp % ID
    tmp = tmp // ID
    ic = tmp % IC
    n = tmp // IC

    x_off = ((n * IC + ic) * ID + id_) * IH * IW + ih * IW + iw
    x_val = tl.load(x_ptr + x_off)  # scalar

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # weight: (IC, OC, KD, KH, KW)
    w_base = ic * OC * KD * KH * KW

    for kd in tl.static_range(0, KD):
        od = id_ * SD + kd - PD
        if (od >= 0) & (od < OD):
            for kh in tl.static_range(0, KH):
                oh = ih * SH + kh - PH
                if (oh >= 0) & (oh < OH):
                    for kw in tl.static_range(0, KW):
                        ow = iw * SW + kw - PW
                        if (ow >= 0) & (ow < OW):
                            w_offs = w_base + oc_offs * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                            w_vals = tl.load(w_ptr + w_offs, mask=oc_mask, other=0.0)
                            contrib = x_val * w_vals
                            out_offs = ((n * OC + oc_offs) * OD + od) * OH * OW + oh * OW + ow
                            tl.atomic_add(out_ptr + out_offs, contrib, mask=oc_mask)


@triton.jit
def init_bias_kernel(
    out_ptr,
    b_ptr,
    N, OC, SPATIAL,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_oc = tl.program_id(1)
    n = pid // ((SPATIAL + BLOCK - 1) // BLOCK)
    blk = pid % ((SPATIAL + BLOCK - 1) // BLOCK)
    
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < SPATIAL
    
    b_val = tl.load(b_ptr + pid_oc)
    out_offs = (n * OC + pid_oc) * SPATIAL + offs
    tl.store(out_ptr + out_offs, b_val, mask=mask)


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
        # Use cudnn conv_transpose3d (typically faster than naive scatter for these shapes)
        x = self.conv_transpose(x)
        x = x.contiguous()

        N, C, D, H, W = x.shape
        k = self.maxpool_kernel_size
        Dp, Hp, Wp = D // k, H // k, W // k

        out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)

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
            grid = (N * C,)
            fused_maxpool_mean_clamp_kernel[grid](
                x, out,
                N, C, D, H, W,
                Dp, Hp, Wp,
                float(self.scale),
                inv_count,
                BLOCK=BLOCK,
                num_warps=4,
            )
        else:
            x = self.maxpool(x)
            x = x.contiguous()
            x = x * self.scale
            x = x.mean(dim=[2, 3, 4], keepdim=True)
            out = torch.clamp(x, 0.0, 1.0)
        return out