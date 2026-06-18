import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_gather_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_hw = tl.program_id(2)

    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    OHW = OH * OW
    hw_mask = hw_offs < OHW
    oh = hw_offs // OW
    ow = hw_offs % OW

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    od = pid_d

    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    for kd in tl.static_range(KD):
        id_num = od + PD - kd
        id_val = id_num // SD
        id_valid = (id_num >= 0) & ((id_num - id_val * SD) == 0) & (id_val >= 0) & (id_val < ID)
        for kh in tl.static_range(KH):
            ih_num = oh + PH - kh
            ih_val = ih_num // SH
            ih_valid = (ih_num >= 0) & ((ih_num - ih_val * SH) == 0) & (ih_val >= 0) & (ih_val < IH)
            for kw in tl.static_range(KW):
                iw_num = ow + PW - kw
                iw_val = iw_num // SW
                iw_valid = (iw_num >= 0) & ((iw_num - iw_val * SW) == 0) & (iw_val >= 0) & (iw_val < IW)
                valid = id_valid & ih_valid & iw_valid

                for ic in range(IC):
                    x_off = ((pid_n * IC + ic) * ID + id_val) * IH * IW + ih_val * IW + iw_val
                    x_val = tl.load(x_ptr + x_off, mask=valid & hw_mask, other=0.0)
                    w_off = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                    acc += x_val[:, None] * w_val[None, :]

    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b_val[None, :]

    out_off = ((pid_n * OC + oc_offs[None, :]) * OD + od) * OH * OW + oh[:, None] * OW + ow[:, None]
    tl.store(out_ptr + out_off, acc, mask=hw_mask[:, None] & oc_mask[None, :])


@triton.jit
def bn_avgpool4_kernel(
    in_ptr, scale_ptr, shift_ptr, out_ptr,
    N, C, ID, IH, IW,
    OD, OH, OW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C * OD * OH * OW
    mask = offs < total

    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    tmp = tmp // OD
    c = tmp % C
    n = tmp // C

    scale = tl.load(scale_ptr + c, mask=mask, other=0.0)
    shift = tl.load(shift_ptr + c, mask=mask, other=0.0)

    acc = tl.zeros((BLOCK,), dtype=tl.float32)

    id_base = od * 4
    ih_base = oh * 4
    iw_base = ow * 4

    for dd in tl.static_range(4):
        for hh in tl.static_range(4):
            for ww in tl.static_range(4):
                id_v = id_base + dd
                ih_v = ih_base + hh
                iw_v = iw_base + ww
                in_off = ((n * C + c) * ID + id_v) * IH * IW + ih_v * IW + iw_v
                v = tl.load(in_ptr + in_off, mask=mask, other=0.0)
                acc += v

    acc = acc * (1.0 / 64.0)
    out = acc * scale + shift
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def fused_conv_t_bn_pool_kernel(
    x_ptr, w_ptr, b_ptr, scale_ptr, shift_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    OD2, OH2, OW2,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # Each program produces one pooled output cell for a tile of (oh2, ow2)
    # at one (n, od2).  Pooled cell covers a 4x4x4 region in conv-transpose output.
    pid_n = tl.program_id(0)
    pid_d2 = tl.program_id(1)
    pid_hw2 = tl.program_id(2)

    hw2_offs = pid_hw2 * BLOCK_HW + tl.arange(0, BLOCK_HW)
    OHW2 = OH2 * OW2
    hw2_mask = hw2_offs < OHW2
    oh2 = hw2_offs // OW2
    ow2 = hw2_offs % OW2

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # 4x4x4 pooling region starts at:
    od_base = pid_d2 * 4
    oh_base = oh2 * 4  # vector
    ow_base = ow2 * 4  # vector

    pool_acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    # Loop over the 64 positions in the pooling window
    for dd in tl.static_range(4):
        od = od_base + dd
        for hh in tl.static_range(4):
            oh = oh_base + hh  # vector
            for ww in tl.static_range(4):
                ow = ow_base + ww  # vector

                # Compute conv-transpose output for (od, oh, ow)
                acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)
                for kd in tl.static_range(KD):
                    id_num = od + PD - kd
                    id_val = id_num // SD
                    id_valid = (id_num >= 0) & ((id_num - id_val * SD) == 0) & (id_val >= 0) & (id_val < ID)
                    for kh in tl.static_range(KH):
                        ih_num = oh + PH - kh
                        ih_val = ih_num // SH
                        ih_valid = (ih_num >= 0) & ((ih_num - ih_val * SH) == 0) & (ih_val >= 0) & (ih_val < IH)
                        for kw in tl.static_range(KW):
                            iw_num = ow + PW - kw
                            iw_val = iw_num // SW
                            iw_valid = (iw_num >= 0) & ((iw_num - iw_val * SW) == 0) & (iw_val >= 0) & (iw_val < IW)
                            valid = id_valid & ih_valid & iw_valid

                            for ic in range(IC):
                                x_off = ((pid_n * IC + ic) * ID + id_val) * IH * IW + ih_val * IW + iw_val
                                x_val = tl.load(x_ptr + x_off, mask=valid & hw2_mask, other=0.0)
                                w_off = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                                acc += x_val[:, None] * w_val[None, :]

                pool_acc += acc

    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    pool_acc += b_val[None, :] * 64.0  # bias added 64 times

    pool_acc = pool_acc * (1.0 / 64.0)

    scale = tl.load(scale_ptr + oc_offs, mask=oc_mask, other=0.0)
    shift = tl.load(shift_ptr + oc_offs, mask=oc_mask, other=0.0)
    pool_acc = pool_acc * scale[None, :] + shift[None, :]

    out_off = ((pid_n * OC + oc_offs[None, :]) * OD2 + pid_d2) * OH2 * OW2 + oh2[:, None] * OW2 + ow2[:, None]
    tl.store(out_ptr + out_off, pool_acc, mask=hw2_mask[:, None] & oc_mask[None, :])


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous()
        bias = self.conv_transpose.bias.contiguous()

        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        SD = SH = SW = self.stride
        PD = PH = PW = self.padding

        OD = (ID - 1) * SD - 2 * PD + KD
        OH = (IH - 1) * SH - 2 * PH + KH
        OW = (IW - 1) * SW - 2 * PW + KW

        BLOCK_OC = 1
        while BLOCK_OC < OC:
            BLOCK_OC *= 2

        if self.training:
            # Fallback path: produce conv_out via gather kernel and then use torch BN + pools
            conv_out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)
            BLOCK_HW = 32
            grid = (N, OD, (OH * OW + BLOCK_HW - 1) // BLOCK_HW)
            conv_transpose3d_gather_kernel[grid](
                x, weight, bias, conv_out,
                N, IC, ID, IH, IW,
                OC, OD, OH, OW,
                KD, KH, KW,
                SD, SH, SW,
                PD, PH, PW,
                BLOCK_OC=BLOCK_OC,
                BLOCK_HW=BLOCK_HW,
                num_warps=4,
                num_stages=2,
            )
            x2 = self.batch_norm(conv_out)
            x2 = F.avg_pool3d(x2, 2)
            x2 = F.avg_pool3d(x2, 2)
            return x2

        running_mean = self.batch_norm.running_mean
        running_var = self.batch_norm.running_var
        gamma = self.batch_norm.weight
        beta = self.batch_norm.bias
        eps = self.batch_norm.eps

        inv_std = torch.rsqrt(running_var + eps)
        scale = (gamma * inv_std).contiguous()
        shift = (beta - running_mean * scale).contiguous()

        OD2 = OD // 4
        OH2 = OH // 4
        OW2 = OW // 4

        if OD2 * 4 != OD or OH2 * 4 != OH or OW2 * 4 != OW:
            # Fall back: compute conv, BN, pool
            conv_out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)
            BLOCK_HW = 32
            grid = (N, OD, (OH * OW + BLOCK_HW - 1) // BLOCK_HW)
            conv_transpose3d_gather_kernel[grid](
                x, weight, bias, conv_out,
                N, IC, ID, IH, IW,
                OC, OD, OH, OW,
                KD, KH, KW,
                SD, SH, SW,
                PD, PH, PW,
                BLOCK_OC=BLOCK_OC,
                BLOCK_HW=BLOCK_HW,
                num_warps=4,
                num_stages=2,
            )
            x2 = conv_out * scale.view(1, -1, 1, 1, 1) + shift.view(1, -1, 1, 1, 1)
            x2 = F.avg_pool3d(x2, 2)
            x2 = F.avg_pool3d(x2, 2)
            return x2

        # Fused path: directly produce pooled output
        out = torch.empty((N, OC, OD2, OH2, OW2), device=x.device, dtype=x.dtype)
        BLOCK_HW = 16
        grid = (N, OD2, (OH2 * OW2 + BLOCK_HW - 1) // BLOCK_HW)
        fused_conv_t_bn_pool_kernel[grid](
            x, weight, bias, scale, shift, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            OD2, OH2, OW2,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            BLOCK_OC=BLOCK_OC,
            BLOCK_HW=BLOCK_HW,
            num_warps=4,
            num_stages=2,
        )
        return out