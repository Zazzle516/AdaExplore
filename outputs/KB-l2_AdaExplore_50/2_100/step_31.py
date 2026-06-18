import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,        # (N, ID, IH, IW, IC)
    w_ptr,        # (IC, KD, KH, KW, OC)
    b_ptr,        # (OC,)
    out_ptr,      # (N, OD, OH, OW, OC)
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    min_value, inv_divisor,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    sp_off = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_max = OD * OH * OW
    sp_mask = sp_off < sp_max

    od = sp_off // (OH * OW)
    rem = sp_off % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    oc_off = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_off < OC

    # bias load
    b = tl.load(b_ptr + oc_off, mask=oc_mask, other=0.0)
    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32) + b[None, :]

    # For each kernel position, find input voxel that contributes
    # output[od, oh, ow] += sum_{kd,kh,kw,ic} input[id, ih, iw, ic] * w[ic, kd, kh, kw, oc]
    # where: od = id*SD - PD + kd  =>  id = (od + PD - kd) / SD
    for kd in tl.static_range(KD):
        id_num = od + PD - kd
        id_v = id_num // SD
        id_ok = (id_num >= 0) & (id_num < ID * SD) & ((id_num % SD) == 0) & (id_v < ID) & (id_v >= 0)
        for kh in tl.static_range(KH):
            ih_num = oh + PH - kh
            ih_v = ih_num // SH
            ih_ok = (ih_num >= 0) & (ih_num < IH * SH) & ((ih_num % SH) == 0) & (ih_v < IH) & (ih_v >= 0)
            for kw in tl.static_range(KW):
                iw_num = ow + PW - kw
                iw_v = iw_num // SW
                iw_ok = (iw_num >= 0) & (iw_num < IW * SW) & ((iw_num % SW) == 0) & (iw_v < IW) & (iw_v >= 0)

                valid = id_ok & ih_ok & iw_ok & sp_mask  # (BLOCK_SP,)

                # input offset base (for each spatial pos): n*ID*IH*IW*IC + id*IH*IW*IC + ih*IW*IC + iw*IC
                in_sp_off = ((pid_n * ID + id_v) * IH + ih_v) * IW + iw_v  # (BLOCK_SP,)
                in_base = in_sp_off * IC  # in IC units offset

                # weight offset base: ic*(KD*KH*KW*OC) + kd*KH*KW*OC + kh*KW*OC + kw*OC + oc
                w_kbase = ((kd * KH + kh) * KW + kw) * OC  # scalar
                # for each ic: w_ptr + ic*KD*KH*KW*OC + w_kbase + oc

                # GEMM-like over IC
                # x_block: (BLOCK_SP, BLOCK_IC), w_block: (BLOCK_IC, BLOCK_OC)
                BLOCK_IC: tl.constexpr = 32
                for ic_start in range(0, IC, BLOCK_IC):
                    ic_off = ic_start + tl.arange(0, BLOCK_IC)
                    ic_mask = ic_off < IC

                    x_addr = in_base[:, None] + ic_off[None, :]
                    x_m = valid[:, None] & ic_mask[None, :]
                    x_vals = tl.load(x_ptr + x_addr, mask=x_m, other=0.0)

                    w_addr = ic_off[:, None] * (KD * KH * KW * OC) + w_kbase + oc_off[None, :]
                    w_m = ic_mask[:, None] & oc_mask[None, :]
                    w_vals = tl.load(w_ptr + w_addr, mask=w_m, other=0.0)

                    acc += tl.dot(x_vals, w_vals, allow_tf32=True)

    # epilogue: clamp + div
    acc = tl.where(acc < min_value, min_value, acc)
    acc = acc * inv_divisor

    # store: output (N, OD, OH, OW, OC)
    out_sp_off = ((pid_n * OD + od) * OH + oh) * OW + ow  # (BLOCK_SP,)
    out_addr = out_sp_off[:, None] * OC + oc_off[None, :]
    out_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_addr, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.min_value = float(min_value)
        self.divisor = float(divisor)

        # Use a real ConvTranspose3d for parameter init compatibility
        conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                  stride=stride, padding=padding)
        # weight: (in_channels, out_channels, kD, kH, kW)
        # transform to (IC, KD, KH, KW, OC) channels-last
        w = conv.weight.detach().clone()
        w_cl = w.permute(0, 2, 3, 4, 1).contiguous()  # (IC, KD, KH, KW, OC)
        self.weight_cl = nn.Parameter(w_cl)
        self.bias = nn.Parameter(conv.bias.detach().clone())

        self.KD = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.KH = kernel_size if isinstance(kernel_size, int) else kernel_size[1]
        self.KW = kernel_size if isinstance(kernel_size, int) else kernel_size[2]
        self.SD = stride if isinstance(stride, int) else stride[0]
        self.SH = stride if isinstance(stride, int) else stride[1]
        self.SW = stride if isinstance(stride, int) else stride[2]
        self.PD = padding if isinstance(padding, int) else padding[0]
        self.PH = padding if isinstance(padding, int) else padding[1]
        self.PW = padding if isinstance(padding, int) else padding[2]

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD, KH, KW = self.KD, self.KH, self.KW
        SD, SH, SW = self.SD, self.SH, self.SW
        PD, PH, PW = self.PD, self.PH, self.PW

        OD = (ID - 1) * SD - 2 * PD + KD
        OH = (IH - 1) * SH - 2 * PH + KH
        OW = (IW - 1) * SW - 2 * PW + KW

        # convert input to channels-last: (N, ID, IH, IW, IC)
        x_cl = x.permute(0, 2, 3, 4, 1).contiguous()

        out_cl = torch.empty((N, OD, OH, OW, OC), device=x.device, dtype=x.dtype)

        BLOCK_OC = 64
        BLOCK_SP = 32

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OD * OH * OW, BLOCK_SP))

        conv_transpose3d_kernel[grid](
            x_cl, self.weight_cl, self.bias, out_cl,
            N, IC, OC,
            ID, IH, IW,
            OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            self.min_value, 1.0 / self.divisor,
            BLOCK_OC=BLOCK_OC,
            BLOCK_SP=BLOCK_SP,
            num_warps=4,
            num_stages=2,
        )

        # convert back to (N, OC, OD, OH, OW)
        out = out_cl.permute(0, 4, 1, 2, 3).contiguous()
        return out