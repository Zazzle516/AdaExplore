import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def scatter_convt3d_kernel(
    x_ptr, w_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, ic, id, ih, iw) -- spread oc across BLOCK_OC
    pid = tl.program_id(0)
    pid_oc = tl.program_id(1)

    # decompose pid into (n, ic, id, ih, iw)
    iw = pid % IW
    tmp = pid // IW
    ih = tmp % IH
    tmp = tmp // IH
    id_ = tmp % ID
    tmp = tmp // ID
    ic = tmp % IC
    n = tmp // IC

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # load input scalar
    x_off = ((n * IC + ic) * ID + id_) * IH * IW + ih * IW + iw
    x_val = tl.load(x_ptr + x_off)

    # base output coords (top-left of kernel scatter)
    od_base = id_ * STRIDE - PAD
    oh_base = ih * STRIDE - PAD
    ow_base = iw * STRIDE - PAD

    # weight layout: (IC, OC, KD, KH, KW) ; w_ptr offset:
    # w[ic, oc, kd, kh, kw]
    w_ic_base = ic * OC * KD * KH * KW

    for kd in tl.static_range(0, KD):
        od = od_base + kd
        if (od >= 0) & (od < OD):
            for kh in tl.static_range(0, KH):
                oh = oh_base + kh
                if (oh >= 0) & (oh < OH):
                    for kw in tl.static_range(0, KW):
                        ow = ow_base + kw
                        if (ow >= 0) & (ow < OW):
                            w_off = w_ic_base + oc_offs * (KD * KH * KW) + kd * KH * KW + kh * KW + kw
                            w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                            contrib = x_val * w_val
                            out_off = ((n * OC + oc_offs) * OD + od) * OH * OW + oh * OW + ow
                            tl.atomic_add(out_ptr + out_off, contrib, mask=oc_mask)


@triton.jit
def fused_bias_epilogue_kernel(
    out_ptr, bias_ptr,
    N, C, S,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total
    c_idx = (offs // S) % C
    x = tl.load(out_ptr + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    y = (2.0 * x + b) * x + x
    tl.store(out_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        # Reuse PyTorch's nn.ConvTranspose3d to match parameter init exactly
        conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                  stride=stride, padding=padding,
                                  output_padding=output_padding)
        # weight shape: (IC, OC, KD, KH, KW)
        self.weight = nn.Parameter(conv.weight.detach().clone())
        self.conv_bias = nn.Parameter(conv.bias.detach().clone())
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        OC = self.out_channels
        OD = (ID - 1) * self.stride - 2 * self.padding + KD + self.output_padding
        OH = (IH - 1) * self.stride - 2 * self.padding + KH + self.output_padding
        OW = (IW - 1) * self.stride - 2 * self.padding + KW + self.output_padding

        # Initialize output with bias broadcast: shape (OC,) -> (1,OC,1,1,1)
        out = self.conv_bias.view(1, OC, 1, 1, 1).expand(N, OC, OD, OH, OW).contiguous()

        # Launch scatter kernel
        BLOCK_OC = 64
        grid = (N * IC * ID * IH * IW, (OC + BLOCK_OC - 1) // BLOCK_OC)
        scatter_convt3d_kernel[grid](
            x, self.weight, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            self.stride, self.padding,
            BLOCK_OC=BLOCK_OC,
            num_warps=2,
        )

        # Fused epilogue: (2x+b)*x + x  in-place
        total = out.numel()
        S = OD * OH * OW
        bias_flat = self.bias.contiguous().view(-1)
        BLOCK_SIZE = 1024
        grid2 = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        fused_bias_epilogue_kernel[grid2](
            out, bias_flat, N, OC, S, total,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=4,
        )
        return out