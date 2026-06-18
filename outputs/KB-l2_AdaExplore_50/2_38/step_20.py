import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_convt_bias_clamp_kernel(
    x_ptr,           # input: (N, IC, ID, IH, IW)
    w_ptr,           # weight: (IC, OC, KD, KH, KW)
    b_ptr,           # bias: (OC,)
    out_ptr,         # output: (N, OC, OD, OH, OW)
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    clamp_min: tl.constexpr, clamp_max: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # grid: (N*OC, OD*OH, ceil(OW/BLOCK_W))
    pid_nc = tl.program_id(0)
    pid_dh = tl.program_id(1)
    pid_w = tl.program_id(2)

    n = pid_nc // OC
    oc = pid_nc % OC

    od = pid_dh // OH
    oh = pid_dh % OH

    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = ow < OW

    # compute the "padded" output coords
    pd = od + PAD  # = id * STRIDE + kd  =>  id = (pd - kd) / STRIDE  (if divisible)
    ph = oh + PAD
    pw = ow + PAD  # vector

    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    # iterate over kernel positions
    for kd in tl.static_range(0, KD):
        id_num = pd - kd
        id_q = id_num // STRIDE
        id_r = id_num - id_q * STRIDE
        valid_d = (id_r == 0) & (id_q >= 0) & (id_q < ID)
        for kh in tl.static_range(0, KH):
            ih_num = ph - kh
            ih_q = ih_num // STRIDE
            ih_r = ih_num - ih_q * STRIDE
            valid_h = (ih_r == 0) & (ih_q >= 0) & (ih_q < IH)
            for kw in tl.static_range(0, KW):
                iw_num = pw - kw
                iw_q = iw_num // STRIDE
                iw_r = iw_num - iw_q * STRIDE
                valid_w = (iw_r == 0) & (iw_q >= 0) & (iw_q < IW)

                spatial_valid = valid_d & valid_h & valid_w & w_mask

                # Loop over input channels
                for ic in range(0, IC):
                    # weight index: w[ic, oc, kd, kh, kw]
                    w_off = ((ic * OC + oc) * KD + kd) * KH * KW + kh * KW + kw
                    wv = tl.load(w_ptr + w_off)

                    # input index: x[n, ic, id_q, ih_q, iw_q]
                    x_base = ((n * IC + ic) * ID + id_q) * IH * IW + ih_q * IW
                    x_off = x_base + iw_q
                    xv = tl.load(x_ptr + x_off, mask=spatial_valid, other=0.0)
                    acc += xv * wv

    bias = tl.load(b_ptr + oc)
    acc = acc + bias
    acc = tl.minimum(tl.maximum(acc, clamp_min), clamp_max)

    out_base = ((n * OC + oc) * OD + od) * OH * OW + oh * OW
    out_off = out_base + ow
    tl.store(out_ptr + out_off, acc, mask=w_mask)


@triton.jit
def softmax_scale_kernel(
    x_ptr, scale_ptr, out_ptr,
    S,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    C = tl.num_programs(1)

    row_start = (b * C + c) * S
    scale = tl.load(scale_ptr + c)

    max_val = -float('inf')
    sum_val = 0.0
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        x = tl.load(x_ptr + row_start + idx, mask=mask, other=-float('inf'))
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(max_val, block_max)
        sum_val = sum_val * tl.exp(max_val - new_max)
        e = tl.exp(x - new_max)
        e = tl.where(mask, e, 0.0)
        sum_val += tl.sum(e, axis=0)
        max_val = new_max

    inv_sum = 1.0 / sum_val
    coef = inv_sum * scale

    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        x = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        y = tl.exp(x - max_val) * coef
        tl.store(out_ptr + row_start + idx, y, mask=mask)


@triton.jit
def avgpool3d_kernel(
    x_ptr, out_ptr,
    N, C, ID, IH, IW,
    OD, OH, OW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * C * OD * OH * OW
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    ow = offs % OW
    t1 = offs // OW
    oh = t1 % OH
    t2 = t1 // OH
    od = t2 % OD
    t3 = t2 // OD
    c = t3 % C
    n = t3 // C

    id0 = od * 2
    ih0 = oh * 2
    iw0 = ow * 2

    base = ((n * C + c) * ID) * IH * IW

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for dd in tl.static_range(0, 2):
        for hh in tl.static_range(0, 2):
            for ww in tl.static_range(0, 2):
                off_in = base + (id0 + dd) * IH * IW + (ih0 + hh) * IW + (iw0 + ww)
                v = tl.load(x_ptr + off_in, mask=mask, other=0.0)
                acc += v
    acc = acc * 0.125

    tl.store(out_ptr + offs, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = pool_kernel_size
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        # ConvTranspose3d weight has shape (in_channels, out_channels, kD, kH, kW)
        conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                  stride=stride, padding=padding,
                                  output_padding=output_padding)
        self.weight = nn.Parameter(conv.weight.detach().clone())
        self.bias = nn.Parameter(conv.bias.detach().clone())
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    def forward(self, x):
        x = x.contiguous()
        N, IC, D, H, W = x.shape

        # AvgPool3d with kernel=2
        PD, PH, PW = D // 2, H // 2, W // 2
        pooled = torch.empty((N, IC, PD, PH, PW), device=x.device, dtype=x.dtype)
        total = N * IC * PD * PH * PW
        BLOCK_AP = 1024
        grid_ap = ((total + BLOCK_AP - 1) // BLOCK_AP,)
        avgpool3d_kernel[grid_ap](
            x, pooled,
            N, IC, D, H, W,
            PD, PH, PW,
            BLOCK=BLOCK_AP,
            num_warps=4, num_stages=2,
        )

        # ConvTranspose3d output shape
        KD = KH = KW = self.kernel_size
        STR = self.stride
        PAD = self.padding
        OPAD = self.output_padding
        OD = (PD - 1) * STR - 2 * PAD + KD + OPAD
        OH = (PH - 1) * STR - 2 * PAD + KH + OPAD
        OW = (PW - 1) * STR - 2 * PAD + KW + OPAD

        OC = self.out_channels

        conv_out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_W = 32  # output W tile
        grid_cv = (N * OC, OD * OH, (OW + BLOCK_W - 1) // BLOCK_W)
        fused_convt_bias_clamp_kernel[grid_cv](
            pooled, self.weight, self.bias, conv_out,
            N, IC, PD, PH, PW,
            OC, OD, OH, OW,
            KD=KD, KH=KH, KW=KW,
            STRIDE=STR, PAD=PAD,
            clamp_min=self.clamp_min, clamp_max=self.clamp_max,
            BLOCK_W=BLOCK_W,
            num_warps=4, num_stages=2,
        )

        # Softmax over spatial dims + scale
        S = OD * OH * OW
        x_flat = conv_out.view(N, OC, S)
        out = torch.empty_like(x_flat)
        scale_flat = self.scale.view(OC).contiguous()

        BLOCK_SM = 4096
        grid_sm = (N, OC)
        softmax_scale_kernel[grid_sm](
            x_flat, scale_flat, out,
            S,
            BLOCK=BLOCK_SM,
            num_warps=8, num_stages=2,
        )

        return out.view(N, OC, OD, OH, OW)