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
    BLOCK_SPATIAL: tl.constexpr,
):
    # one program per (n, oc, spatial_block)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_s = tl.program_id(2)

    spatial = OD * OH * OW
    offs = pid_s * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    mask = offs < spatial

    od = offs // (OH * OW)
    rem = offs % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    acc = tl.zeros((BLOCK_SPATIAL,), dtype=tl.float32)

    # for each output (od,oh,ow), iterate over (ic, kd, kh, kw)
    # input index: id = (od + PD - kd) / SD, must be divisible and in range
    for ic in range(0, IC):
        for kd in range(0, KD):
            for kh in range(0, KH):
                for kw in range(0, KW):
                    id_num = od + PD - kd
                    ih_num = oh + PH - kh
                    iw_num = ow + PW - kw

                    id_q = id_num // SD
                    ih_q = ih_num // SH
                    iw_q = iw_num // SW

                    valid = (id_num >= 0) & (ih_num >= 0) & (iw_num >= 0)
                    valid = valid & ((id_num - id_q * SD) == 0)
                    valid = valid & ((ih_num - ih_q * SH) == 0)
                    valid = valid & ((iw_num - iw_q * SW) == 0)
                    valid = valid & (id_q >= 0) & (id_q < ID)
                    valid = valid & (ih_q >= 0) & (ih_q < IH)
                    valid = valid & (iw_q >= 0) & (iw_q < IW)
                    valid = valid & mask

                    x_off = (((pid_n * IC + ic) * ID + id_q) * IH + ih_q) * IW + iw_q
                    x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)

                    # weight: (IC, OC, KD, KH, KW)
                    w_off = (((ic * OC + pid_oc) * KD + kd) * KH + kh) * KW + kw
                    w_val = tl.load(w_ptr + w_off)

                    acc += x_val * w_val

    b_val = tl.load(b_ptr + pid_oc)
    acc += b_val

    out_off = ((pid_n * OC + pid_oc) * OD + od) * OH * OW + oh * OW + ow
    out_ptr_base = out_ptr + ((pid_n * OC + pid_oc) * spatial)
    tl.store(out_ptr_base + offs, acc, mask=mask)


@triton.jit
def fused_bn_avgpool4_kernel(
    x_ptr, scale_ptr, shift_ptr, out_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    BLOCK: tl.constexpr,
):
    # one program per (n, c), tiles over output spatial
    pid_nc = tl.program_id(0)
    pid_s = tl.program_id(1)
    n = pid_nc // C
    c = pid_nc % C

    out_spatial = OD * OH * OW
    offs = pid_s * BLOCK + tl.arange(0, BLOCK)
    mask = offs < out_spatial

    ow = offs % OW
    t1 = offs // OW
    oh = t1 % OH
    od = t1 // OH

    d0 = od * 4
    h0 = oh * 4
    w0 = ow * 4

    acc = tl.zeros((BLOCK,), dtype=tl.float32)

    base = (n * C + c) * D
    for dd in range(0, 4):
        for hh in range(0, 4):
            for ww in range(0, 4):
                d_idx = d0 + dd
                h_idx = h0 + hh
                w_idx = w0 + ww
                in_off = ((base + d_idx) * H + h_idx) * W + w_idx
                v = tl.load(x_ptr + in_off, mask=mask, other=0.0)
                acc += v

    acc = acc * (1.0 / 64.0)

    # scale/shift broadcast per channel (loaded once)
    s = tl.load(scale_ptr + c)
    sh = tl.load(shift_ptr + c)
    acc = acc * s + sh

    out_base = (n * C + c) * out_spatial
    tl.store(out_ptr + out_base + offs, acc, mask=mask)


def conv_transpose3d_triton(x, weight, bias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    SD = SH = SW = stride
    PD = PH = PW = padding

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_SPATIAL = 128
    spatial = OD * OH * OW
    grid = (N, OC, (spatial + BLOCK_SPATIAL - 1) // BLOCK_SPATIAL)

    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_SPATIAL=BLOCK_SPATIAL,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous()
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias

        # use PyTorch's optimized conv_transpose3d (cuDNN)
        y = F.conv_transpose3d(x, weight, bias, stride=self.stride, padding=self.padding)

        bn = self.batch_norm
        if self.training:
            # compute per-channel mean/var across (N,D,H,W) using PyTorch for now
            N, C, D, H, W = y.shape
            # update running stats and produce scale/shift matching F.batch_norm(training=True)
            y_flat = y.reshape(N, C, -1)
            mean = y_flat.mean(dim=(0, 2))
            # biased variance for normalization, unbiased for running_var
            var_biased = y_flat.var(dim=(0, 2), unbiased=False)
            with torch.no_grad():
                n_elem = N * D * H * W
                if n_elem > 1:
                    var_unbiased = var_biased * (n_elem / (n_elem - 1))
                else:
                    var_unbiased = var_biased
                bn.running_mean.mul_(1 - bn.momentum).add_(mean.detach(), alpha=bn.momentum)
                bn.running_var.mul_(1 - bn.momentum).add_(var_unbiased.detach(), alpha=bn.momentum)
                bn.num_batches_tracked.add_(1)
            scale = bn.weight / torch.sqrt(var_biased + bn.eps)
            shift = bn.bias - mean * scale

            OD, OH, OW = D // 4, H // 4, W // 4
            out = torch.empty((N, C, OD, OH, OW), device=y.device, dtype=y.dtype)
            out_spatial = OD * OH * OW
            BLOCK = 256
            grid = (N * C, (out_spatial + BLOCK - 1) // BLOCK)
            fused_bn_avgpool4_kernel[grid](
                y, scale.contiguous(), shift.contiguous(), out,
                N, C, D, H, W,
                OD, OH, OW,
                BLOCK=BLOCK, num_warps=4, num_stages=2,
            )
            return out
        else:
            # fold BN into scale/shift
            eps = bn.eps
            var = bn.running_var
            mean = bn.running_mean
            gamma = bn.weight
            beta = bn.bias
            scale = gamma / torch.sqrt(var + eps)
            shift = beta - mean * scale

            N, C, D, H, W = y.shape
            OD, OH, OW = D // 4, H // 4, W // 4
            out = torch.empty((N, C, OD, OH, OW), device=y.device, dtype=y.dtype)
            out_spatial = OD * OH * OW
            BLOCK = 256
            grid = (N * C, (out_spatial + BLOCK - 1) // BLOCK)
            fused_bn_avgpool4_kernel[grid](
                y, scale.contiguous(), shift.contiguous(), out,
                N, C, D, H, W,
                OD, OH, OW,
                BLOCK=BLOCK, num_warps=4, num_stages=2,
            )
            return out