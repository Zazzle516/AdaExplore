import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_clamp_div_kernel(
    x_ptr, out_ptr, n_elements,
    min_value, inv_divisor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = tl.where(x < min_value, min_value, x)
    x = x * inv_divisor
    tl.store(out_ptr + offsets, x, mask=mask)


def fused_clamp_div(x, min_value, divisor):
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 4096
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_clamp_div_kernel[grid](
        x, out, n,
        float(min_value), float(1.0 / divisor),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=3,
    )
    return out


@triton.jit
def scatter_convtranspose3d_kernel(
    x_ptr,        # [N, IC, D, H, W]
    w_ptr,        # [IC, OC, KD, KH, KW]
    b_ptr,        # [OC]
    out_ptr,      # [N, OC, OD, OH, OW]
    N, IC, D, H, W,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    min_value, inv_divisor,
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, ic, d, h, w) input element ... too many.
    # better: one program per (n, output_voxel) computing direct conv
    pass


# Direct gather-based ConvTranspose3d using equivalent convolution formulation.
# For ConvTranspose3d with stride S, pad P, kernel K:
#   out[n, oc, od, oh, ow] = sum over ic, kd, kh, kw of
#     x[n, ic, id, ih, iw] * weight[ic, oc, kd, kh, kw]
#   where  od + P = id*S + kd  =>  id = (od + P - kd) / S, with mod==0
# So for each output position, we iterate kd,kh,kw and check divisibility.

@triton.jit
def conv_transpose3d_fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, D, H, W,
    OC, OD, OH, OW,
    min_value, inv_divisor,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    OHW = OH * OW
    OSP = OD * OH * OW

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    sp_mask = sp_offs < OSP
    oc_mask = oc_offs < OC

    # decompose sp_offs into (od, oh, ow)
    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    # accumulator [BLOCK_SP, BLOCK_OC]
    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # Iterate over kernel positions and ic
    for kd in tl.static_range(0, KD):
        id_num = od + PAD - kd
        id_val = id_num // STRIDE
        id_ok = (id_num % STRIDE == 0) & (id_val >= 0) & (id_val < D)
        for kh in tl.static_range(0, KH):
            ih_num = oh + PAD - kh
            ih_val = ih_num // STRIDE
            ih_ok = (ih_num % STRIDE == 0) & (ih_val >= 0) & (ih_val < H)
            for kw in tl.static_range(0, KW):
                iw_num = ow + PAD - kw
                iw_val = iw_num // STRIDE
                iw_ok = (iw_num % STRIDE == 0) & (iw_val >= 0) & (iw_val < W)
                pos_ok = id_ok & ih_ok & iw_ok & sp_mask  # [BLOCK_SP]

                # input offset base for n, *, id_val, ih_val, iw_val
                # x layout: [N, IC, D, H, W]
                # We will loop over IC
                base_x = pid_n * IC * D * H * W + id_val * H * W + ih_val * W + iw_val  # [BLOCK_SP]

                # weight offset: w[ic, oc, kd, kh, kw]
                # layout: [IC, OC, KD, KH, KW]
                w_kpos = kd * KH * KW + kh * KW + kw  # scalar

                for ic in range(0, IC):
                    x_off = base_x + ic * D * H * W
                    x_val = tl.load(x_ptr + x_off, mask=pos_ok, other=0.0)  # [BLOCK_SP]

                    w_off = ic * OC * KD * KH * KW + oc_offs * KD * KH * KW + w_kpos  # [BLOCK_OC]
                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    acc += x_val[:, None] * w_val[None, :]

    # add bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[None, :]

    # clamp + div
    acc = tl.where(acc < min_value, min_value, acc)
    acc = acc * inv_divisor

    # store [N, OC, OD, OH, OW]
    out_off = pid_n * OC * OSP + oc_offs[None, :] * OSP + sp_offs[:, None]
    out_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.min_value = float(min_value)
        self.divisor = float(divisor)

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous()  # [IC, OC, KD, KH, KW]
        bias = self.conv_transpose.bias.contiguous()      # [OC]

        N, IC, D, H, W = x.shape
        OC = self.out_channels
        K = self.kernel_size
        S = self.stride
        P = self.padding

        OD = (D - 1) * S - 2 * P + K
        OH = (H - 1) * S - 2 * P + K
        OW = (W - 1) * S - 2 * P + K

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 64
        BLOCK_SP = 64
        OSP = OD * OH * OW
        grid = (N, (OC + BLOCK_OC - 1) // BLOCK_OC, (OSP + BLOCK_SP - 1) // BLOCK_SP)

        conv_transpose3d_fused_kernel[grid](
            x, weight, bias, out,
            N, IC, D, H, W,
            OC, OD, OH, OW,
            self.min_value, 1.0 / self.divisor,
            KD=K, KH=K, KW=K,
            STRIDE=S, PAD=P,
            BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
            num_warps=4,
            num_stages=2,
        )
        return out