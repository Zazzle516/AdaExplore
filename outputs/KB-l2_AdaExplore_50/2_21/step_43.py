import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_bias_scale_sigmoid_gn_kernel(
    x_ptr,            # input [N, IC, H_in, W_in]
    w_ptr,            # conv weight [OC, IC, KH, KW]
    cb_ptr,           # conv bias [OC]
    bias_ptr,         # extra bias [OC]
    scale_ptr,        # scale [OC]
    gn_w_ptr,         # GN weight [OC]
    gn_b_ptr,         # GN bias [OC]
    out_ptr,          # output [N, OC, H_out, W_out]
    N, IC, H_in, W_in,
    OC, H_out, W_out,
    G, CPG,
    eps,
    KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    CPG_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    HW = H_out * W_out
    group_size = CPG_C * HW

    c_off = tl.arange(0, CPG_C)            # [CPG]
    c_global = g * CPG_C + c_off            # [CPG]

    # Preload params for this group's channels
    cb = tl.load(cb_ptr + c_global).to(tl.float32)       # [CPG]
    eb = tl.load(bias_ptr + c_global).to(tl.float32)     # [CPG]
    sc = tl.load(scale_ptr + c_global).to(tl.float32)    # [CPG]
    gw = tl.load(gn_w_ptr + c_global).to(tl.float32)     # [CPG]
    gb = tl.load(gn_b_ptr + c_global).to(tl.float32)     # [CPG]

    bias_combined = (cb + eb) * sc                       # [CPG]

    sum_val = 0.0
    sum_sq = 0.0

    n_tiles = (HW + BLOCK_HW - 1) // BLOCK_HW

    # We'll write the intermediate sigmoid output into out_ptr first (temporary)
    # Then second-pass normalize in-place.
    for t in range(0, n_tiles):
        hw_offs = t * BLOCK_HW + tl.arange(0, BLOCK_HW)     # [BLOCK_HW]
        hw_mask = hw_offs < HW
        oh = hw_offs // W_out
        ow = hw_offs % W_out

        # Accumulator: [CPG, BLOCK_HW]
        acc = tl.zeros([CPG_C, BLOCK_HW], dtype=tl.float32)

        for ic in tl.static_range(0, IC_C):
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    ih = oh + kh        # [BLOCK_HW]
                    iw = ow + kw        # [BLOCK_HW]
                    x_off = n * IC * H_in * W_in + ic * H_in * W_in + ih * W_in + iw
                    x_val = tl.load(x_ptr + x_off, mask=hw_mask, other=0.0).to(tl.float32)  # [BLOCK_HW]
                    # weight for all CPG output channels, this ic, kh, kw
                    w_off = c_global * IC * KH * KW + ic * KH * KW + kh * KW + kw  # [CPG]
                    w_val = tl.load(w_ptr + w_off).to(tl.float32)  # [CPG]
                    acc += w_val[:, None] * x_val[None, :]

        # Apply bias + scale + sigmoid
        acc = acc + bias_combined[:, None]
        acc = tl.sigmoid(acc)
        # mask
        acc = tl.where(hw_mask[None, :], acc, 0.0)

        # accumulate sum, sum_sq
        sum_val += tl.sum(acc).to(tl.float32)
        sum_sq += tl.sum(acc * acc).to(tl.float32)

        # store temp
        out_off = n * OC * HW + c_global[:, None] * HW + hw_offs[None, :]
        tl.store(out_ptr + out_off, acc, mask=hw_mask[None, :])

    gs = (CPG_C * HW).to(tl.float32)
    mean = sum_val / gs
    var = sum_sq / gs - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: read temp, normalize, write back
    for t in range(0, n_tiles):
        hw_offs = t * BLOCK_HW + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offs < HW
        out_off = n * OC * HW + c_global[:, None] * HW + hw_offs[None, :]
        v = tl.load(out_ptr + out_off, mask=hw_mask[None, :], other=0.0)
        v = (v - mean) * rstd * gw[:, None] + gb[:, None]
        tl.store(out_ptr + out_off, v, mask=hw_mask[None, :])


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, IC, H_in, W_in = x.shape
        KH = KW = self.kernel_size
        OC = self.out_channels
        H_out = H_in - KH + 1
        W_out = W_in - KW + 1
        G = self.num_groups
        CPG = OC // G

        out = torch.empty(N, OC, H_out, W_out, device=x.device, dtype=x.dtype)

        w = self.conv.weight.contiguous()
        cb = self.conv.bias.contiguous()
        bias_flat = self.bias.view(-1).contiguous()
        scale_flat = self.scale.view(-1).contiguous()
        gn_w = self.group_norm.weight.contiguous()
        gn_b = self.group_norm.bias.contiguous()
        eps = self.group_norm.eps

        BLOCK_HW = 512
        grid = (N * G,)
        fused_conv_bias_scale_sigmoid_gn_kernel[grid](
            x, w, cb, bias_flat, scale_flat, gn_w, gn_b, out,
            N, IC, H_in, W_in,
            OC, H_out, W_out,
            G, CPG, eps,
            KH=KH, KW=KW,
            IC_C=IC, CPG_C=CPG,
            BLOCK_HW=BLOCK_HW,
            num_warps=8, num_stages=2,
        )
        return out