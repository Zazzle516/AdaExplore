import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def scatter_convtranspose3d_kernel(
    x_ptr, w_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, id, ih, iw)
    pid = tl.program_id(0)
    iw = pid % IW
    pid = pid // IW
    ih = pid % IH
    pid = pid // IH
    id_ = pid % ID
    n = pid // ID

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Loop over input channels and kernel positions
    for ic in range(0, IC):
        x_val = tl.load(x_ptr + n * (IC * ID * IH * IW) + ic * (ID * IH * IW) + id_ * (IH * IW) + ih * IW + iw)
        for kd in range(0, KD):
            od = id_ * SD - PD + kd
            for kh in range(0, KH):
                oh = ih * SH - PH + kh
                for kw in range(0, KW):
                    ow = iw * SW - PW + kw
                    in_bounds = (od >= 0) & (od < OD) & (oh >= 0) & (oh < OH) & (ow >= 0) & (ow < OW)
                    # weight: [IC, OC, KD, KH, KW]
                    w_offset = ic * (OC * KD * KH * KW) + oc_offs * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w_val = tl.load(w_ptr + w_offset, mask=oc_mask, other=0.0)
                    contrib = x_val * w_val
                    out_offset = n * (OC * OD * OH * OW) + oc_offs * (OD * OH * OW) + od * (OH * OW) + oh * OW + ow
                    tl.atomic_add(out_ptr + out_offset, contrib, mask=oc_mask & in_bounds)


@triton.jit
def fused_post_kernel(
    inp_ptr, bias_ptr, out_ptr,
    conv_bias_ptr,
    N, OC, OD, OH, OW,
    POD, POH, POW,
    scale1, scale2,
    BLOCK: tl.constexpr,
):
    # one program per (n, oc, pooled_d_tile)
    # We'll launch flat over all output elements after pool
    pid = tl.program_id(0)
    total = N * OC * POD * POH * POW
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    tmp = offs
    pw = tmp % POW
    tmp = tmp // POW
    ph = tmp % POH
    tmp = tmp // POH
    pd = tmp % POD
    tmp = tmp // POD
    oc = tmp % OC
    n = tmp // OC

    # avg pool 2x2x2: sum over 8 elements and divide by 8
    od0 = pd * 2
    oh0 = ph * 2
    ow0 = pw * 2

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    conv_b = tl.load(conv_bias_ptr + oc, mask=mask, other=0.0)
    for ddz in range(0, 2):
        for ddy in range(0, 2):
            for ddx in range(0, 2):
                od = od0 + ddz
                oh = oh0 + ddy
                ow = ow0 + ddx
                off = n * (OC * OD * OH * OW) + oc * (OD * OH * OW) + od * (OH * OW) + oh * OW + ow
                v = tl.load(inp_ptr + off, mask=mask, other=0.0)
                acc += v + conv_b

    acc = acc * (1.0 / 8.0)
    acc = acc * scale1
    b = tl.load(bias_ptr + oc, mask=mask, other=0.0)
    acc = acc + b
    acc = acc * scale2

    tl.store(out_ptr + offs, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Match ConvTranspose3d's parameter init
        conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.weight = nn.Parameter(conv.weight.detach().clone())
        self.conv_bias = nn.Parameter(conv.bias.detach().clone())

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

        conv_out = torch.zeros((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

        BLOCK_OC = triton.next_power_of_2(OC)
        if BLOCK_OC < 16:
            BLOCK_OC = 16

        grid = (N * ID * IH * IW,)
        scatter_convtranspose3d_kernel[grid](
            x, self.weight, conv_out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            BLOCK_OC=BLOCK_OC,
        )

        # Pooled dims
        POD = OD // 2
        POH = OH // 2
        POW = OW // 2

        out = torch.empty((N, OC, POD, POH, POW), device=x.device, dtype=torch.float32)

        total = N * OC * POD * POH * POW
        BLOCK = 256
        grid2 = ((total + BLOCK - 1) // BLOCK,)
        bias_flat = self.bias.view(-1).contiguous()
        fused_post_kernel[grid2](
            conv_out, bias_flat, out,
            self.conv_bias,
            N, OC, OD, OH, OW,
            POD, POH, POW,
            float(self.scale1.item()), float(self.scale2.item()),
            BLOCK=BLOCK,
        )
        return out