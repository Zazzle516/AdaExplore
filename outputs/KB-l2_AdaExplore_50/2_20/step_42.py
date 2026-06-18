import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_scatter_kernel(
    x_ptr,           # [N, IC, ID, IH, IW]
    w_ptr,           # [IC, OC, KD, KH, KW]
    out_ptr,         # [N, OC, OD, OH, OW]
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    KD, KH, KW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # Each program: one (n, id, ih, iw) and a tile of OC, looping over kernel positions
    pid_n = tl.program_id(0)
    pid_spatial = tl.program_id(1)
    pid_oc = tl.program_id(2)

    iw = pid_spatial % IW
    ih = (pid_spatial // IW) % IH
    id_ = pid_spatial // (IW * IH)

    if id_ >= ID:
        return

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    # Load all input values for this (n, id, ih, iw) across all IC
    # x[n, ic, id, ih, iw] for ic in [0, IC)
    # We'll loop over IC blocks
    x_base = pid_n * (IC * ID * IH * IW) + id_ * (IH * IW) + ih * IW + iw
    # x stride for ic: ID*IH*IW

    # Accumulator for OC tile output contributions per kernel pos
    # Output position: od = id*sd - pd + kd, oh = ih*sh - ph + kh, ow = iw*sw - pw + kw
    # For each (kd, kh, kw): add sum_ic( x[n,ic,id,ih,iw] * w[ic, oc, kd, kh, kw] ) to out[n, oc, od, oh, ow]
    # Since each input position scatters to distinct outputs (no overlap from same input), no atomics needed
    # (different inputs may scatter to same output -> need atomics)

    out_base_n_oc = pid_n * (OC * OD * OH * OW)

    for kd in range(KD):
        od = id_ * stride_d - pad_d + kd
        if (od >= 0) & (od < OD):
            for kh in range(KH):
                oh = ih * stride_h - pad_h + kh
                if (oh >= 0) & (oh < OH):
                    for kw in range(KW):
                        ow = iw * stride_w - pad_w + kw
                        if (ow >= 0) & (ow < OW):
                            # Compute sum over IC: x[n,ic,id,ih,iw] * w[ic, :, kd, kh, kw]
                            acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)
                            for ic_start in range(0, IC, BLOCK_IC):
                                offs_ic = ic_start + tl.arange(0, BLOCK_IC)
                                mask_ic = offs_ic < IC
                                # Load x: [BLOCK_IC]
                                x_ptrs = x_ptr + x_base + offs_ic * (ID * IH * IW)
                                x_vals = tl.load(x_ptrs, mask=mask_ic, other=0.0)
                                # Load weight: w[ic, oc, kd, kh, kw]
                                # weight shape: [IC, OC, KD, KH, KW]
                                # stride: oc has KD*KH*KW, ic has OC*KD*KH*KW
                                w_ptrs = (w_ptr
                                          + offs_ic[:, None] * (OC * KD * KH * KW)
                                          + offs_oc[None, :] * (KD * KH * KW)
                                          + kd * (KH * KW) + kh * KW + kw)
                                w_vals = tl.load(w_ptrs,
                                                 mask=mask_ic[:, None] & mask_oc[None, :],
                                                 other=0.0)
                                acc += tl.sum(x_vals[:, None] * w_vals, axis=0)

                            out_ptrs = (out_ptr + out_base_n_oc
                                        + offs_oc * (OD * OH * OW)
                                        + od * (OH * OW) + oh * OW + ow)
                            tl.atomic_add(out_ptrs, acc, mask=mask_oc)


@triton.jit
def fused_bias_epilogue_kernel(
    x_ptr,      # in/out [N, C, D, H, W] - already has conv bias
    bias_ptr,   # [C]
    n_elements,
    C: tl.constexpr, DHW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c_idx = (offsets // DHW) % C
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    out = (2.0 * x + b) * x + x
    tl.store(x_ptr + offsets, out, mask=mask)


def conv_transpose3d_triton(x, weight, conv_bias, stride, padding):
    """
    x: [N, IC, ID, IH, IW]
    weight: [IC, OC, KD, KH, KW]
    conv_bias: [OC] or None
    """
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    sd, sh, sw = stride
    pd, ph, pw = padding

    OD = (ID - 1) * sd - 2 * pd + KD + (1 if (sd > 1) else 0)  # output_padding
    OH = (IH - 1) * sh - 2 * ph + KH + (1 if (sh > 1) else 0)
    OW = (IW - 1) * sw - 2 * pw + KW + (1 if (sw > 1) else 0)

    # Initialize output with conv bias broadcast (or zero)
    if conv_bias is not None:
        out = conv_bias.view(1, OC, 1, 1, 1).expand(N, OC, OD, OH, OW).contiguous()
    else:
        out = torch.zeros((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 64
    BLOCK_IC = 32

    grid = (N, ID * IH * IW, triton.cdiv(OC, BLOCK_OC))

    conv_transpose3d_scatter_kernel[grid](
        x, weight, out,
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        sd, sh, sw,
        pd, ph, pw,
        BLOCK_OC=BLOCK_OC,
        BLOCK_IC=BLOCK_IC,
        num_warps=4,
        num_stages=2,
    )
    return out


def fused_epilogue_inplace(x: torch.Tensor, bias: torch.Tensor):
    bias_flat = bias.contiguous().view(-1)
    n_elements = x.numel()
    N, C, D, H, W = x.shape
    DHW = D * H * W
    BLOCK_SIZE = 4096
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_bias_epilogue_kernel[grid](
        x, bias_flat, n_elements, C, DHW,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=4,
    )
    return x


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding, output_padding)
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)

    def forward(self, x):
        # Use torch's optimized conv_transpose3d (cuDNN) - hard to beat
        x = self.conv_transpose(x)
        return fused_epilogue_inplace(x, self.bias)