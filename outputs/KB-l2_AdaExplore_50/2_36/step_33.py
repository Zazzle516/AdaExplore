import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['IC', 'OC', 'IH', 'IW', 'KH', 'KW', 'PARITY'],
)
@triton.jit
def conv_transpose_parity_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    PH_OFF: tl.constexpr,  # parity offset for h: oh % SH
    PW_OFF: tl.constexpr,  # parity offset for w: ow % SW
    OH_P: tl.constexpr,    # number of output rows of this parity (OH+SH-1-PH_OFF)//SH
    OW_P: tl.constexpr,    # OW per parity
    KH_P: tl.constexpr,    # valid kh's for this parity = ceil((KH - ((PH+PH_OFF)%SH))/SH)... we just iterate compile-time set
    KW_P: tl.constexpr,
    PARITY: tl.constexpr,  # combined parity id for autotune key
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Specialize by (oh%SH, ow%SW) parity. For a given parity (PH_OFF, PW_OFF),
    the valid (kh, kw) taps are those for which (PH + PH_OFF - kh) % SH == 0
    and (PW + PW_OFF - kw) % SW == 0. This makes K = IC * KH_P * KW_P, much smaller.
    """
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    sp_offs = pid_sp * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    oc_offs = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)

    # decode sp into (oh_p, ow_p) in parity grid; then map to full oh, ow.
    oh_p = sp_offs // OW_P
    ow_p = sp_offs % OW_P
    oh = oh_p * SH + PH_OFF
    ow = ow_p * SW + PW_OFF
    sp_mask = sp_offs < (OH_P * OW_P)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K_total = IC * KH_P * KW_P

    # First valid kh: smallest kh s.t. (PH + PH_OFF - kh) % SH == 0, kh in [0, KH)
    # i.e., kh ≡ (PH + PH_OFF) (mod SH). Let kh0 = (PH + PH_OFF) % SH.
    KH0: tl.constexpr = (PH + PH_OFF) % SH
    KW0: tl.constexpr = (PW + PW_OFF) % SW

    for k_start in range(0, K_total, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K_total

        ic = k_offs // (KH_P * KW_P)
        rem = k_offs % (KH_P * KW_P)
        khi = rem // KW_P    # index within valid kh's
        kwi = rem % KW_P
        kh = KH0 + khi * SH
        kw = KW0 + kwi * SW

        # ih = (oh + PH - kh) / SH, guaranteed divisible
        ih = (oh[:, None] + PH - kh[None, :]) // SH
        iw = (ow[:, None] + PW - kw[None, :]) // SW
        valid = (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)

        x_off = pid_n * (IC * IH * IW) + ic[None, :] * (IH * IW) + ih * IW + iw
        x_mask = valid & sp_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

        # W is (IC, OC, KH, KW)
        w_off = (ic[:, None] * (OC * KH * KW)
                 + oc_offs[None, :] * (KH * KW)
                 + kh[:, None] * KW
                 + kw[:, None])
        w_mask = k_mask[:, None] & oc_mask[None, :]
        w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

        acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b_vals[None, :]

    # Store into full output (N, OC, OH, OW) at the parity positions
    acc_t = tl.trans(acc)  # [BLOCK_N, BLOCK_M]
    out_sp_off = oh * OW + ow  # [BLOCK_M]
    out_off = (pid_n * (OC * OH * OW)
               + oc_offs[:, None] * (OH * OW)
               + out_sp_off[None, :])
    mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc_t, mask=mask)


@triton.jit
def min_sum_gelu_bias_kernel(
    inp_ptr,
    bias_ptr,
    out_ptr,
    N, OC, OH, OW,
    BLOCK_H: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_w = tl.program_id(1)

    oc_range = tl.arange(0, BLOCK_OC)
    oh_range = tl.arange(0, BLOCK_H)

    sum_acc = 0.0

    for h_start in range(0, OH, BLOCK_H):
        oh_offs = h_start + oh_range
        h_mask = oh_offs < OH

        min_vals = tl.full((BLOCK_H,), float('inf'), dtype=tl.float32)
        for c_start in range(0, OC, BLOCK_OC):
            oc_offs = c_start + oc_range
            c_mask = oc_offs < OC
            offs = (pid_n * OC * OH * OW
                    + oc_offs[:, None] * (OH * OW)
                    + oh_offs[None, :] * OW
                    + pid_w)
            mask = c_mask[:, None] & h_mask[None, :]
            vals = tl.load(inp_ptr + offs, mask=mask, other=float('inf'))
            tile_min = tl.min(vals, axis=0)
            min_vals = tl.minimum(min_vals, tile_min)

        min_vals = tl.where(h_mask, min_vals, 0.0)
        sum_acc += tl.sum(min_vals, axis=0)

    x = sum_acc
    gelu = 0.5 * x * (1.0 + tl.erf(x / 1.4142135623730951))
    b = tl.load(bias_ptr)
    res = gelu + b

    out_off = pid_n * OW + pid_w
    tl.store(out_ptr + out_off, res)


def conv_transpose2d_triton(x, weight, bias_param, stride, padding, output_padding):
    N, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w
    SH, SW = stride, stride
    PH, PW = padding, padding
    OH = (IH - 1) * SH - 2 * PH + KH + output_padding
    OW = (IW - 1) * SW - 2 * PW + KW + output_padding

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.float32)

    # Launch one grid per parity
    for ph_off in range(SH):
        for pw_off in range(SW):
            # Number of output positions of this parity
            # oh runs over values in [0, OH) with oh % SH == ph_off
            OH_P = (OH - ph_off + SH - 1) // SH if OH > ph_off else 0
            OW_P = (OW - pw_off + SW - 1) // SW if OW > pw_off else 0
            if OH_P == 0 or OW_P == 0:
                continue

            # Number of valid kh's: kh in [0, KH) with kh % SH == (PH+ph_off)%SH
            kh0 = (PH + ph_off) % SH
            KH_P = (KH - kh0 + SH - 1) // SH if KH > kh0 else 0
            kw0 = (PW + pw_off) % SW
            KW_P = (KW - kw0 + SW - 1) // SW if KW > kw0 else 0
            if KH_P == 0 or KW_P == 0:
                continue

            parity_id = ph_off * SW + pw_off

            grid = lambda META, OHP=OH_P, OWP=OW_P: (
                N,
                triton.cdiv(OC, META['BLOCK_N']),
                triton.cdiv(OHP * OWP, META['BLOCK_M']),
            )
            conv_transpose_parity_kernel[grid](
                x, weight, bias_param, out,
                N, IC, IH, IW,
                OC, OH, OW,
                KH, KW, SH, SW, PH, PW,
                ph_off, pw_off,
                OH_P, OW_P, KH_P, KW_P,
                parity_id,
            )

    return out


def min_sum_gelu_bias_triton(inp, bias):
    N, OC, OH, OW = inp.shape
    out = torch.empty((N, 1, 1, OW), device=inp.device, dtype=torch.float32)
    BLOCK_OC = 128
    BLOCK_H = 32
    grid = (N, OW)
    min_sum_gelu_bias_kernel[grid](
        inp, bias, out,
        N, OC, OH, OW,
        BLOCK_H=BLOCK_H, BLOCK_OC=BLOCK_OC,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv_transpose.weight.contiguous().cuda()
        b = self.conv_transpose.bias.contiguous().cuda()
        conv_out = conv_transpose2d_triton(x, w, b, self.stride, self.padding, self.output_padding)
        bias_scalar = self.bias.contiguous().view(-1)[0:1].cuda()
        out = min_sum_gelu_bias_triton(conv_out, bias_scalar)
        return out