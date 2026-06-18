import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, od, oh, ow) tile across OC
    pid_n = tl.program_id(0)
    pid_spatial = tl.program_id(1)
    pid_oc = tl.program_id(2)

    od = pid_spatial // (OH * OW)
    rem = pid_spatial % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # For each input position, find kernel position that contributes to (od, oh, ow)
    # out[od,oh,ow] = sum over (id,ih,iw,kd,kh,kw) where
    #   od = id*SD - PD + kd  =>  kd = od + PD - id*SD
    # So for each id: kd = od + PD - id*SD, must be in [0, KD)
    for ic in range(0, IC):
        for kd in range(0, KD):
            id_val = (od + PD - kd)
            id_q = id_val // SD
            id_r = id_val - id_q * SD
            valid_d = (id_r == 0) & (id_q >= 0) & (id_q < ID)
            for kh in range(0, KH):
                ih_val = (oh + PH - kh)
                ih_q = ih_val // SH
                ih_r = ih_val - ih_q * SH
                valid_h = (ih_r == 0) & (ih_q >= 0) & (ih_q < IH)
                for kw in range(0, KW):
                    iw_val = (ow + PW - kw)
                    iw_q = iw_val // SW
                    iw_r = iw_val - iw_q * SW
                    valid_w = (iw_r == 0) & (iw_q >= 0) & (iw_q < IW)
                    valid = valid_d & valid_h & valid_w
                    if valid:
                        x_off = ((pid_n * IC + ic) * ID + id_q) * IH * IW + ih_q * IW + iw_q
                        x_val = tl.load(x_ptr + x_off)
                        # weight shape: (IC, OC, KD, KH, KW)
                        w_off = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                        w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                        acc += x_val * w_val

    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b_val

    out_off = ((pid_n * OC + oc_offs) * OD + od) * OH * OW + oh * OW + ow
    tl.store(out_ptr + out_off, acc, mask=oc_mask)


@triton.jit
def fused_pool_bias_scale_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    scale_combined,  # scale1^3 / 8 * scale2... actually scale1 already applied? we'll apply scale1*scale2 here
    BLOCK: tl.constexpr,
):
    # x has shape (N, C, D, H, W) - the conv output (without scale1 applied)
    # output shape: (N, C, OD, OH, OW) where OD=D//2 etc.
    # avg_pool then *scale1, +bias, *scale2
    # acc = (sum of 8 values) / 8 * scale1, then +bias, then *scale2
    pid = tl.program_id(0)
    total = N * C * OD * OH * OW
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    tmp = tmp // OD
    c = tmp % C
    n = tmp // C

    d0 = od * 2
    h0 = oh * 2
    w0 = ow * 2

    base = ((n * C + c) * D + d0) * H * W + h0 * W + w0

    s = tl.zeros((BLOCK,), dtype=tl.float32)
    for dd in range(0, 2):
        for hh in range(0, 2):
            for ww in range(0, 2):
                off = base + dd * H * W + hh * W + ww
                v = tl.load(x_ptr + off, mask=mask, other=0.0)
                s += v

    s = s / 8.0
    s = s * scale_combined  # scale1
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    s = s + b
    # scale2 applied separately
    tl.store(out_ptr + offs, s, mask=mask)


@triton.jit
def scale_kernel(x_ptr, out_ptr, n_elements, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x * scale, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Match nn.ConvTranspose3d initialization
        ct = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.conv_weight = nn.Parameter(ct.weight.data.clone())
        self.conv_bias = nn.Parameter(ct.bias.data.clone())

        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        SD = SH = SW = self.stride
        PD = PH = PW = self.padding
        OC = self.out_channels

        OD = (ID - 1) * SD - 2 * PD + KD
        OH = (IH - 1) * SH - 2 * PH + KH
        OW = (IW - 1) * SW - 2 * PW + KW

        conv_out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

        BLOCK_OC = 16
        grid = (N, OD * OH * OW, triton.cdiv(OC, BLOCK_OC))
        conv_transpose3d_kernel[grid](
            x, self.conv_weight, self.conv_bias, conv_out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            BLOCK_OC=BLOCK_OC,
        )

        # avg_pool with kernel 2 -> output shape OD//2, OH//2, OW//2
        POD = OD // 2
        POH = OH // 2
        POW = OW // 2
        pooled = torch.empty((N, OC, POD, POH, POW), device=x.device, dtype=torch.float32)

        total = N * OC * POD * POH * POW
        BLOCK = 256
        grid2 = (triton.cdiv(total, BLOCK),)
        scale_combined = float(self.scale1.item())
        fused_pool_bias_scale_kernel[grid2](
            conv_out, self.bias, pooled,
            N, OC, OD, OH, OW,
            POD, POH, POW,
            scale_combined,
            BLOCK=BLOCK,
        )

        # Apply scale2
        out = torch.empty_like(pooled)
        n_elem = pooled.numel()
        grid3 = (triton.cdiv(n_elem, 1024),)
        scale_kernel[grid3](pooled, out, n_elem, float(self.scale2.item()), BLOCK=1024)
        return out