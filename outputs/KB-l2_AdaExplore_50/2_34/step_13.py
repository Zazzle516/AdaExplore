import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose3d_scatter_kernel(
    x_ptr, w_ptr, out_ptr,
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # grid: (N * ID * IH * IW, OC // BLOCK_OC)
    pid_spatial = tl.program_id(0)
    pid_oc = tl.program_id(1)

    iw = pid_spatial % IW
    tmp = pid_spatial // IW
    ih = tmp % IH
    tmp = tmp // IH
    id_ = tmp % ID
    n = tmp // ID

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    ic_offs = tl.arange(0, BLOCK_IC)

    # Load input vector for all IC at this spatial location
    # x layout: (N, IC, ID, IH, IW)
    x_base = ((n * IC) * ID + id_) * IH * IW + ih * IW + iw
    # We need x[n, ic, id, ih, iw] for ic in [0, IC)
    # stride between ic = ID*IH*IW
    
    # Accumulate per (kd, kh, kw): out_pos = (id*SD - PD + kd, ...)
    # Compute outer product sum over IC: acc[oc] = sum_ic x[ic] * w[ic, oc, kd, kh, kw]

    for kd in range(KD):
        od = id_ * SD - PD + kd
        if (od >= 0) & (od < OD):
            for kh in range(KH):
                oh = ih * SH - PH + kh
                if (oh >= 0) & (oh < OH):
                    for kw in range(KW):
                        ow = iw * SW - PW + kw
                        if (ow >= 0) & (ow < OW):
                            acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)
                            for ic_start in range(0, IC, BLOCK_IC):
                                ic_idx = ic_start + ic_offs
                                ic_mask = ic_idx < IC
                                # x[n, ic, id, ih, iw]
                                x_ptrs = x_ptr + x_base + ic_idx * (ID * IH * IW)
                                x_vals = tl.load(x_ptrs, mask=ic_mask, other=0.0)  # (BLOCK_IC,)
                                # w[ic, oc, kd, kh, kw], stride: oc -> KD*KH*KW, ic -> OC*KD*KH*KW
                                w_offset = (kd * KH + kh) * KW + kw
                                w_ptrs = (w_ptr
                                          + ic_idx[:, None] * (OC * KD * KH * KW)
                                          + oc_offs[None, :] * (KD * KH * KW)
                                          + w_offset)
                                w_mask = ic_mask[:, None] & oc_mask[None, :]
                                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)
                                acc += tl.sum(x_vals[:, None] * w_vals, axis=0)
                            
                            out_offset = (((n * OC + oc_offs) * OD + od) * OH + oh) * OW + ow
                            tl.atomic_add(out_ptr + out_offset, acc, mask=oc_mask)


def conv_transpose3d_triton(x, weight, bias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    assert IC == IC_w
    SD = SH = SW = stride
    PD = PH = PW = padding

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    if bias is not None:
        out = bias.view(1, OC, 1, 1, 1).expand(N, OC, OD, OH, OW).contiguous()
    else:
        out = torch.zeros((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    x_contig = x.contiguous()
    w_contig = weight.contiguous()

    BLOCK_OC = 64
    BLOCK_IC = 32
    grid = (N * ID * IH * IW, triton.cdiv(OC, BLOCK_OC))
    
    conv_transpose3d_scatter_kernel[grid](
        x_contig, w_contig, out,
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_OC=BLOCK_OC,
        BLOCK_IC=BLOCK_IC,
        num_warps=4,
        num_stages=2,
    )
    return out


@triton.jit
def layernorm_gelu_scale_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    M, C,
    eps, scaling_factor,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C

    x_row_ptr = x_ptr + row * C
    out_row_ptr = out_ptr + row * C

    x = tl.load(x_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
    mean = sum_x / C
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * w + b
    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))
    gelu = gelu * scaling_factor

    tl.store(out_row_ptr + offs, gelu, mask=mask)


def layernorm_gelu_scale(x, weight, bias, eps, scaling_factor):
    # LayerNorm normalizes over last dim. We want to normalize over the OC channel.
    # x shape: (N, OC, D, H, W). Need to permute so OC is last.
    N, OC, D, H, W = x.shape
    x_perm = x.permute(0, 2, 3, 4, 1).contiguous()  # (N, D, H, W, OC)
    M = N * D * H * W
    out = torch.empty_like(x_perm)
    BLOCK_C = triton.next_power_of_2(OC)
    grid = (M,)
    layernorm_gelu_scale_kernel[grid](
        x_perm, out, weight, bias,
        M, OC,
        eps, scaling_factor,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out.permute(0, 4, 1, 2, 3).contiguous()


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.eps = eps
        self.scaling_factor = scaling_factor
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = conv_transpose3d_triton(
            x, self.conv_transpose.weight, self.conv_transpose.bias,
            self.stride, self.padding
        )
        x = layernorm_gelu_scale(
            x, self.layer_norm.weight, self.layer_norm.bias,
            self.eps, self.scaling_factor
        )
        return x