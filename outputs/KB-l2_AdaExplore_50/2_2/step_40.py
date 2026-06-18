import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Per-parity direct kernel:
# For stride=2, pad=1, kernel=3, output_pad=1, OH=2*IH, OW=2*IW.
# Split output by parity (oh%2, ow%2). For each parity p=(ph,pw), the valid
# (kh,kw) values are uniquely determined as: kh = (oh + pad) % stride giving
# a single contributing (kh,kw) pair per parity for stride=2.
#
# Specifically with PAD=1, STRIDE=2:
#   ih_num = oh + 1 - kh; we need ih_num % 2 == 0 and ih = ih_num/2 in [0,IH)
#   For oh even: (oh+1-kh) even => kh odd => kh=1.   ih = (oh+1-1)/2 = oh/2
#   For oh odd : kh even => kh in {0,2}.
#     kh=0: ih = (oh+1)/2;   valid if (oh+1)/2 < IH  (i.e., oh < 2*IH-1)
#     kh=2: ih = (oh-1)/2;   valid if (oh-1)/2 >= 0  (i.e., oh >= 1)
#
# So we have effectively 4 parities (oh%2, ow%2). For (0,0): one (kh,kw)=(1,1) tap.
# For (0,1): kw in {0,2}, kh=1 => 2 taps. For (1,0): kh in {0,2}, kw=1 => 2 taps.
# For (1,1): (kh,kw) in {0,2}x{0,2} => 4 taps.
#
# We'll implement a generic per-parity loop: for each parity, iterate over the
# valid (kh,kw) taps (computed at host side as static lists), accumulate over IC
# using GEMM. This avoids the stride parity branching inside the kernel.
#
# Each program computes a tile of shape [BLOCK_OC, BLOCK_SP_PER_PARITY]
# where SP_PER_PARITY = IH*IW (since OH*OW/4 = IH*IW).
# Output spatial coords for this parity: (oh = 2*ih + ph, ow = 2*iw + pw)
# Input ih,iw is read directly (no division by stride needed).


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64,  'BLOCK_SP': 128, 'BLOCK_IC': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64,  'BLOCK_SP': 128, 'BLOCK_IC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64,  'BLOCK_IC': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64,  'BLOCK_IC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64,  'BLOCK_SP': 256, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 128, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64,  'BLOCK_SP': 64,  'BLOCK_IC': 64}, num_warps=4, num_stages=3),
    ],
    key=['OC', 'IC', 'IH', 'IW'],
)
@triton.jit
def conv_transpose2d_parity_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    PH: tl.constexpr, PW: tl.constexpr,
    # Up to 4 (kh,kw) taps; unused entries set KH_i = -1 to disable.
    KH0: tl.constexpr, KW0: tl.constexpr, DIH0: tl.constexpr, DIW0: tl.constexpr,
    KH1: tl.constexpr, KW1: tl.constexpr, DIH1: tl.constexpr, DIW1: tl.constexpr,
    KH2: tl.constexpr, KW2: tl.constexpr, DIH2: tl.constexpr, DIW2: tl.constexpr,
    KH3: tl.constexpr, KW3: tl.constexpr, DIH3: tl.constexpr, DIW3: tl.constexpr,
    NTAPS: tl.constexpr,
    KW_TOTAL: tl.constexpr,
    SCALE: tl.constexpr,
    INV_SCALE: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_sp = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_n  = tl.program_id(2)

    SP_PAR = IH * IW  # number of output positions in this parity

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    oc_mask = oc_offs < OC
    sp_mask = sp_offs < SP_PAR

    # Decompose sp_offs into (ih, iw) since each parity tile is IH x IW
    ih = sp_offs // IW
    iw = sp_offs - ih * IW

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    x_batch_off = pid_n * (IC * IH * IW)

    # Loop over IC in blocks. Inside, accumulate contributions from all taps.
    for ic_start in range(0, IC, BLOCK_IC):
        ic_offs = ic_start + tl.arange(0, BLOCK_IC)
        ic_mask = ic_offs < IC

        # Tap 0
        if NTAPS >= 1:
            ih0 = ih + DIH0
            iw0 = iw + DIW0
            v0 = (ih0 >= 0) & (ih0 < IH) & (iw0 >= 0) & (iw0 < IW) & sp_mask
            x_addr0 = x_batch_off + ic_offs[:, None] * (IH * IW) + ih0[None, :] * IW + iw0[None, :]
            x_tile0 = tl.load(x_ptr + x_addr0, mask=ic_mask[:, None] & v0[None, :], other=0.0)
            w_addr0 = ic_offs[None, :] * (OC * KW_TOTAL) + oc_offs[:, None] * KW_TOTAL + KH0 * 3 + KW0
            w_tile0 = tl.load(w_ptr + w_addr0, mask=oc_mask[:, None] & ic_mask[None, :], other=0.0)
            acc += tl.dot(w_tile0, x_tile0, allow_tf32=True)

        if NTAPS >= 2:
            ih1 = ih + DIH1
            iw1 = iw + DIW1
            v1 = (ih1 >= 0) & (ih1 < IH) & (iw1 >= 0) & (iw1 < IW) & sp_mask
            x_addr1 = x_batch_off + ic_offs[:, None] * (IH * IW) + ih1[None, :] * IW + iw1[None, :]
            x_tile1 = tl.load(x_ptr + x_addr1, mask=ic_mask[:, None] & v1[None, :], other=0.0)
            w_addr1 = ic_offs[None, :] * (OC * KW_TOTAL) + oc_offs[:, None] * KW_TOTAL + KH1 * 3 + KW1
            w_tile1 = tl.load(w_ptr + w_addr1, mask=oc_mask[:, None] & ic_mask[None, :], other=0.0)
            acc += tl.dot(w_tile1, x_tile1, allow_tf32=True)

        if NTAPS >= 3:
            ih2 = ih + DIH2
            iw2 = iw + DIW2
            v2 = (ih2 >= 0) & (ih2 < IH) & (iw2 >= 0) & (iw2 < IW) & sp_mask
            x_addr2 = x_batch_off + ic_offs[:, None] * (IH * IW) + ih2[None, :] * IW + iw2[None, :]
            x_tile2 = tl.load(x_ptr + x_addr2, mask=ic_mask[:, None] & v2[None, :], other=0.0)
            w_addr2 = ic_offs[None, :] * (OC * KW_TOTAL) + oc_offs[:, None] * KW_TOTAL + KH2 * 3 + KW2
            w_tile2 = tl.load(w_ptr + w_addr2, mask=oc_mask[:, None] & ic_mask[None, :], other=0.0)
            acc += tl.dot(w_tile2, x_tile2, allow_tf32=True)

        if NTAPS >= 4:
            ih3 = ih + DIH3
            iw3 = iw + DIW3
            v3 = (ih3 >= 0) & (ih3 < IH) & (iw3 >= 0) & (iw3 < IW) & sp_mask
            x_addr3 = x_batch_off + ic_offs[:, None] * (IH * IW) + ih3[None, :] * IW + iw3[None, :]
            x_tile3 = tl.load(x_ptr + x_addr3, mask=ic_mask[:, None] & v3[None, :], other=0.0)
            w_addr3 = ic_offs[None, :] * (OC * KW_TOTAL) + oc_offs[:, None] * KW_TOTAL + KH3 * 3 + KW3
            w_tile3 = tl.load(w_ptr + w_addr3, mask=oc_mask[:, None] & ic_mask[None, :], other=0.0)
            acc += tl.dot(w_tile3, x_tile3, allow_tf32=True)

    # Bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[:, None]

    # clamp(0,1) -> *scale -> clamp(0,1) -> /scale
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    acc = acc * SCALE
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    acc = acc * INV_SCALE

    # Write to output[n, oc, oh, ow] where oh = 2*ih + PH, ow = 2*iw + PW
    oh = 2 * ih + PH
    ow = 2 * iw + PW
    out_off = pid_n * (OC * OH * OW) + oc_offs[:, None] * (OH * OW) + oh[None, :] * OW + ow[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


def _compute_taps(ph, pw, stride, pad, kh_max, kw_max):
    """For each output parity (ph,pw) determine list of (kh, kw, dih, diw) taps.
    where dih = (ph + pad - kh) // stride, diw = (pw + pad - kw) // stride, and
    valid require (ph+pad-kh) % stride == 0 and same for w.
    Then for output (oh,ow) = (stride*ih+ph, stride*iw+pw), input position is
    (ih + dih, iw + diw). We require this in [0,IH) etc., handled at runtime.
    """
    taps = []
    for kh in range(kh_max):
        num_h = ph + pad - kh
        if num_h % stride != 0:
            continue
        dih = num_h // stride
        for kw in range(kw_max):
            num_w = pw + pad - kw
            if num_w % stride != 0:
                continue
            diw = num_w // stride
            taps.append((kh, kw, dih, diw))
    return taps


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding,
                                                  output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous()
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        stride = self.stride
        pad = self.padding
        OH = (IH - 1) * stride - 2 * pad + KH + self.output_padding
        OW = (IW - 1) * stride - 2 * pad + KW + self.output_padding

        fused_bias = (self.conv_transpose.bias + self.bias.view(-1)).contiguous()
        weight = self.conv_transpose.weight.contiguous()  # [IC, OC, KH, KW]

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        scale = float(self.scaling_factor)
        inv_scale = 1.0 / scale

        # For stride=2 case (the configured one), each parity has up to 4 taps.
        # We launch 4 kernel calls, one per parity (ph, pw).
        SP_PAR = IH * IW
        for ph in range(stride):
            for pw in range(stride):
                taps = _compute_taps(ph, pw, stride, pad, KH, KW)
                ntaps = len(taps)
                if ntaps == 0:
                    continue
                # Pad to 4 taps
                padded = list(taps) + [(0, 0, 0, 0)] * (4 - ntaps)
                (kh0, kw0, dih0, diw0) = padded[0]
                (kh1, kw1, dih1, diw1) = padded[1]
                (kh2, kw2, dih2, diw2) = padded[2]
                (kh3, kw3, dih3, diw3) = padded[3]

                grid = lambda meta: (triton.cdiv(SP_PAR, meta['BLOCK_SP']),
                                     triton.cdiv(OC, meta['BLOCK_OC']),
                                     N)

                conv_transpose2d_parity_kernel[grid](
                    x, weight, fused_bias, out,
                    N, IC, IH, IW,
                    OC, OH, OW,
                    ph, pw,
                    kh0, kw0, dih0, diw0,
                    kh1, kw1, dih1, diw1,
                    kh2, kw2, dih2, diw2,
                    kh3, kw3, dih3, diw3,
                    ntaps,
                    KW,
                    scale, inv_scale,
                )

        return out