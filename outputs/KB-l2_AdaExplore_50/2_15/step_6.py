import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_scatter_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    scale_ptr,  # OC: gamma/sqrt(var+eps)
    shift_ptr,  # OC: beta - mean*gamma/sqrt(var+eps)
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, id, ih, iw) input voxel; iterates over OC tile and (kd,kh,kw)
    pid_n = tl.program_id(0)
    pid_spatial = tl.program_id(1)
    pid_oc = tl.program_id(2)

    iw = pid_spatial % IW
    ih = (pid_spatial // IW) % IH
    id_ = pid_spatial // (IW * IH)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Load input value across IC
    # x[n, ic, id, ih, iw]
    # We'll loop over IC and accumulate? Actually scatter-add: for each IC, output += x[ic] * w[ic, oc, kd, kh, kw]
    # We compute outer product over OC tile

    base_d = id_ * SD - PD
    base_h = ih * SH - PH
    base_w = iw * SW - PW

    # bias
    if b_ptr is not None:
        bias_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    else:
        bias_vals = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    for kd in range(KD):
        od = base_d + kd
        for kh in range(KH):
            oh = base_h + kh
            for kw in range(KW):
                ow = base_w + kw
                in_bounds = (od >= 0) & (od < OD) & (oh >= 0) & (oh < OH) & (ow >= 0) & (ow < OW)
                if in_bounds:
                    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)
                    for ic in range(IC):
                        x_val = tl.load(x_ptr + pid_n * IC * ID * IH * IW + ic * ID * IH * IW + id_ * IH * IW + ih * IW + iw)
                        # weight layout: [IC, OC, KD, KH, KW]
                        w_offs = ic * OC * KD * KH * KW + oc_offs * KD * KH * KW + kd * KH * KW + kh * KW + kw
                        w_vals = tl.load(w_ptr + w_offs, mask=oc_mask, other=0.0)
                        acc += x_val * w_vals
                    # this is partial - if multiple input voxels scatter to same output, need atomic
                    out_offs = pid_n * OC * OD * OH * OW + oc_offs * OD * OH * OW + od * OH * OW + oh * OW + ow
                    tl.atomic_add(out_ptr + out_offs, acc, mask=oc_mask)


@triton.jit
def add_bias_bn_kernel(
    out_ptr, bias_ptr, scale_ptr, shift_ptr,
    N, OC, S,
    BLOCK_S: tl.constexpr,
):
    # one program per (n, oc, block of S)
    pid_nc = tl.program_id(0)
    pid_s = tl.program_id(1)
    n = pid_nc // OC
    oc = pid_nc % OC
    offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = offs < S
    base = n * OC * S + oc * S
    x = tl.load(out_ptr + base + offs, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + oc)
    scale = tl.load(scale_ptr + oc)
    shift = tl.load(shift_ptr + oc)
    y = (x + bias) * scale + shift
    tl.store(out_ptr + base + offs, y, mask=mask)


@triton.jit
def sub_mean_kernel(
    x_ptr, out_ptr,
    S,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * S
    sum_val = tl.zeros((), dtype=tl.float32)
    for off in range(0, S, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < S
        x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
    mean = sum_val / S
    for off in range(0, S, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < S
        x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)
        tl.store(out_ptr + row_start + offs, x - mean, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.stride = stride if isinstance(stride, tuple) else (stride,) * 3
        self.padding = padding if isinstance(padding, tuple) else (padding,) * 3

    def forward(self, x):
        # Use torch's optimized conv_transpose3d, then fused BN affine, then mean-subtract
        x = self.conv_transpose(x)
        # BN at eval: fold running stats and affine
        if not self.training:
            bn = self.batch_norm
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            shift = bn.bias - bn.running_mean * scale
            x = x * scale.view(1, -1, 1, 1, 1) + shift.view(1, -1, 1, 1, 1)
        else:
            x = self.batch_norm(x)

        N, C, D, H, W = x.shape
        S = D * H * W
        x_flat = x.contiguous().view(N * C, S)
        out = torch.empty_like(x_flat)
        grid = (N * C,)
        sub_mean_kernel[grid](x_flat, out, S, BLOCK_SIZE=1024, num_warps=4)
        return out.view(N, C, D, H, W)