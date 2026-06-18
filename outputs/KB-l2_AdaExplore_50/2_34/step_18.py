import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# ---------------------------------------------------------------------------
# ConvTranspose3d as scatter-add into NDHWC output, with bias added once.
# Each program handles one input element (n, ic, d, h, w) and scatters its
# contribution across the kernel volume into all output channels.
# Output layout is (N, D_out, H_out, W_out, OC) — contiguous along OC so that
# the subsequent LayerNorm-over-channels has contiguous reductions.
# ---------------------------------------------------------------------------

@triton.jit
def _convT3d_scatter_kernel(
    x_ptr, w_ptr, out_ptr,
    N, IC, D, H, W,
    OC, KD, KH, KW,
    OD, OH, OW,
    SD, SH, SW,
    PD, PH, PW,
    BLOCK_OC: tl.constexpr,
):
    # program ids
    pid_n   = tl.program_id(0)         # batch index
    pid_dhw = tl.program_id(1)         # flat (d,h,w) input spatial index
    pid_ic  = tl.program_id(2)         # input channel

    d = pid_dhw // (H * W)
    rem = pid_dhw - d * (H * W)
    h = rem // W
    w = rem - h * W

    # load input value (scalar)
    x_off = ((pid_n * IC + pid_ic) * D + d) * H * W + h * W + w
    x_val = tl.load(x_ptr + x_off)

    offs_oc = tl.arange(0, BLOCK_OC)
    oc_mask = offs_oc < OC

    # weight stride layout: (IC, OC, KD, KH, KW)
    w_base_ic = pid_ic * OC * KD * KH * KW

    # base output position (before adding kd, kh, kw)
    d_base = d * SD - PD
    h_base = h * SH - PH
    w_base = w * SW - PW

    for kd in range(0, KD):
        od = d_base + kd
        if (od >= 0) & (od < OD):
            for kh in range(0, KH):
                oh = h_base + kh
                if (oh >= 0) & (oh < OH):
                    for kw in range(0, KW):
                        ow = w_base + kw
                        if (ow >= 0) & (ow < OW):
                            w_off = w_base_ic + offs_oc * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                            w_vals = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                            contrib = x_val * w_vals
                            out_off = (((pid_n * OD + od) * OH + oh) * OW + ow) * OC + offs_oc
                            tl.atomic_add(out_ptr + out_off, contrib, mask=oc_mask)


# ---------------------------------------------------------------------------
# Fused LayerNorm (over channels, last dim) + GELU + scaling.
# Input/output layout: (N*OD*OH*OW, C)
# ---------------------------------------------------------------------------

@triton.jit
def _ln_gelu_scale_kernel(
    x_ptr, y_ptr, gamma_ptr, beta_ptr,
    C, eps, scale,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C

    x_row = x_ptr + row * C
    x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / C
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(gamma_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(beta_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    xn = xc * rstd
    y = xn * g + b

    inv_sqrt2 = 0.70710678118654752440
    y_gelu = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))
    out = y_gelu * scale

    tl.store(y_ptr + row * C + offs, out, mask=mask)


def _conv_transpose3d_ndhwc(x, weight, bias, stride, padding):
    """
    x: (N, IC, D, H, W) contiguous
    weight: (IC, OC, KD, KH, KW)
    bias: (OC,) or None
    Returns: (N, OD, OH, OW, OC) contiguous along last dim.
    """
    N, IC, D, H, W = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    assert IC == IC_w
    SD, SH, SW = stride
    PD, PH, PW = padding

    OD = (D - 1) * SD - 2 * PD + KD
    OH = (H - 1) * SH - 2 * PH + KH
    OW = (W - 1) * SW - 2 * PW + KW

    # Initialize output with bias broadcast across spatial positions.
    if bias is not None:
        # broadcast bias (OC,) over (N, OD, OH, OW, OC)
        out = bias.view(1, 1, 1, 1, OC).expand(N, OD, OH, OW, OC).contiguous()
    else:
        out = torch.zeros((N, OD, OH, OW, OC), device=x.device, dtype=x.dtype)

    BLOCK_OC = triton.next_power_of_2(OC)
    grid = (N, D * H * W, IC)

    _convT3d_scatter_kernel[grid](
        x, weight, out,
        N, IC, D, H, W,
        OC, KD, KH, KW,
        OD, OH, OW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_OC=BLOCK_OC,
        num_warps=4,
        num_stages=2,
    )
    return out, (N, OD, OH, OW, OC)


def _ln_gelu_scale_ndhwc(x_ndhwc, gamma, beta, eps, scale):
    """
    x_ndhwc: (N, OD, OH, OW, C) contiguous
    Returns: (N, C, OD, OH, OW)
    """
    N, OD, OH, OW, C = x_ndhwc.shape
    x2d = x_ndhwc.reshape(-1, C)
    M = x2d.shape[0]
    out2d = torch.empty_like(x2d)
    BLOCK_C = triton.next_power_of_2(C)
    num_warps = 4 if BLOCK_C <= 256 else 8
    _ln_gelu_scale_kernel[(M,)](
        x2d, out2d, gamma, beta,
        C, eps, scale,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
    )
    out_ndhwc = out2d.view(N, OD, OH, OW, C)
    # permute back to (N, C, OD, OH, OW)
    return out_ndhwc.permute(0, 4, 1, 2, 3).contiguous()


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 bias=True, eps=1e-5, scaling_factor=1.0):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels,
                                                  kernel_size, stride=stride,
                                                  padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.eps = float(eps)
        self.scaling_factor = float(scaling_factor)
        self.in_channels = in_channels
        self.out_channels = out_channels
        # normalize kernel_size/stride/padding to triples
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        if isinstance(stride, int):
            self.stride = (stride, stride, stride)
        else:
            self.stride = tuple(stride)
        if isinstance(padding, int):
            self.padding = (padding, padding, padding)
        else:
            self.padding = tuple(padding)

    def forward(self, x):
        x = x.contiguous()
        weight = self.conv_transpose.weight  # (IC, OC, KD, KH, KW)
        bias = self.conv_transpose.bias

        out_ndhwc, _ = _conv_transpose3d_ndhwc(
            x, weight, bias, self.stride, self.padding
        )

        out = _ln_gelu_scale_ndhwc(
            out_ndhwc, self.layer_norm.weight, self.layer_norm.bias,
            self.eps, self.scaling_factor
        )
        return out