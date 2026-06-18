import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_scatter_kernel(
    x_ptr, w_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    OD2, OH2, OW2,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # One program per (n, id, ih, iw_tile). Scatter-adds outer product into pooled output.
    pid_n = tl.program_id(0)
    pid_dh = tl.program_id(1)
    pid_w = tl.program_id(2)

    iid = pid_dh // IH
    ih = pid_dh % IH

    iw_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = iw_offs < IW  # shape (BLOCK_W,)

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Load x[n, ic, iid, ih, iw_offs] for all ic, accumulate ic loop inside kd/kh/kw.
    # Compute outer product contributions, scatter via atomic_add into pooled output.

    # Precompute output spatial positions
    # od = iid * SD - PD + kd
    # oh = ih * SH - PH + kh
    # ow = iw_offs * SW - PW + kw  (vector)

    for kd in tl.static_range(KD):
        od = iid * SD - PD + kd
        od_valid = (od >= 0) & (od < OD)
        od2 = od // 4
        for kh in tl.static_range(KH):
            oh = ih * SH - PH + kh
            oh_valid = (oh >= 0) & (oh < OH)
            oh2 = oh // 4
            for kw in tl.static_range(KW):
                ow = iw_offs * SW - PW + kw  # (BLOCK_W,)
                ow_valid = (ow >= 0) & (ow < OW)
                ow2 = ow // 4
                valid_v = od_valid & oh_valid & ow_valid & w_mask  # (BLOCK_W,)

                # Compute contribution: acc[w, oc] = sum_ic x[n,ic,iid,ih,iw] * w[ic,oc,kd,kh,kw]
                acc = tl.zeros((BLOCK_W, BLOCK_OC), dtype=tl.float32)
                for ic in range(IC):
                    x_off = ((pid_n * IC + ic) * ID + iid) * IH * IW + ih * IW + iw_offs
                    x_val = tl.load(x_ptr + x_off, mask=w_mask, other=0.0)
                    w_off = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                    acc += x_val[:, None] * w_val[None, :]

                # scatter-add into pooled output[(n, oc, od2, oh2, ow2)] with weight 1/64
                acc = acc * (1.0 / 64.0)
                out_off = ((pid_n * OC + oc_offs[None, :]) * OD2 + od2) * OH2 * OW2 + oh2 * OW2 + ow2[:, None]
                store_mask = valid_v[:, None] & oc_mask[None, :]
                tl.atomic_add(out_ptr + out_off, acc, mask=store_mask)


@triton.jit
def epilogue_kernel(
    inout_ptr, scale_ptr, shift_ptr,
    N, C, OD2, OH2, OW2,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C * OD2 * OH2 * OW2
    mask = offs < total

    tmp = offs // OW2
    tmp = tmp // OH2
    tmp = tmp // OD2
    c = tmp % C

    scale = tl.load(scale_ptr + c, mask=mask, other=0.0)
    shift = tl.load(shift_ptr + c, mask=mask, other=0.0)
    v = tl.load(inout_ptr + offs, mask=mask, other=0.0)
    out = v * scale + shift
    tl.store(inout_ptr + offs, out, mask=mask)


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
        # bias absorbed into shift: shift = (beta - running_mean*scale) + bias*scale
        shift = (beta - running_mean * scale + bias * scale).contiguous()

        OD2 = OD // 4
        OH2 = OH // 4
        OW2 = OW // 4

        if OD2 * 4 != OD or OH2 * 4 != OH or OW2 * 4 != OW:
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
            scale_b = (gamma * inv_std).contiguous()
            shift_b = (beta - running_mean * scale_b).contiguous()
            x2 = conv_out * scale_b.view(1, -1, 1, 1, 1) + shift_b.view(1, -1, 1, 1, 1)
            x2 = F.avg_pool3d(x2, 2)
            x2 = F.avg_pool3d(x2, 2)
            return x2

        # Scatter-based fused conv-transpose + avg-pool(4)
        out = torch.zeros((N, OC, OD2, OH2, OW2), device=x.device, dtype=x.dtype)

        BLOCK_W = 32 if IW >= 32 else 16
        grid = (N, ID * IH, (IW + BLOCK_W - 1) // BLOCK_W)
        conv_transpose3d_scatter_kernel[grid](
            x, weight, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            OD2, OH2, OW2,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            BLOCK_OC=BLOCK_OC,
            BLOCK_W=BLOCK_W,
            num_warps=4,
            num_stages=2,
        )

        # Epilogue: apply scale, shift in-place
        total = N * OC * OD2 * OH2 * OW2
        BLOCK = 256
        grid2 = ((total + BLOCK - 1) // BLOCK,)
        epilogue_kernel[grid2](
            out, scale, shift,
            N, OC, OD2, OH2, OW2,
            BLOCK=BLOCK,
            num_warps=4,
        )

        return out