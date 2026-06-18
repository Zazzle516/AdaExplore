import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_gather_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    inv_scale,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (OH * OW)

    oh = sp_offs // OW
    ow = sp_offs % OW

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # Iterate kernel positions
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            # Determine valid input position:
            # x_unpadded_h = oh + PAD_H - kh; must be divisible by STRIDE_H
            ih_num = oh + PAD_H - kh
            iw_num = ow + PAD_W - kw

            ih = ih_num // STRIDE_H
            iw = iw_num // STRIDE_W

            valid_h = (ih_num % STRIDE_H == 0) & (ih >= 0) & (ih < IH)
            valid_w = (iw_num % STRIDE_W == 0) & (iw >= 0) & (iw < IW)
            valid = valid_h & valid_w & sp_mask  # [BLOCK_SP]

            # Reduce over IC
            for ic in range(0, IC):
                # x[n, ic, ih, iw]
                x_offset = ((pid_n * IC + ic) * IH + ih) * IW + iw
                x_val = tl.load(x_ptr + x_offset, mask=valid, other=0.0)  # [BLOCK_SP]

                # w[ic, oc, kh, kw]
                w_offset = ((ic * OC + oc_offs) * KH + kh) * KW + kw
                w_val = tl.load(w_ptr + w_offset, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                acc += w_val[:, None] * x_val[None, :]

    # Bias
    bias_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias_val[:, None]

    # clamp [0,1] -> *scale -> clamp [0,1] -> /scale
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    # x * s clamped to [0,1], then /s == clamp(acc, 0, 1/s)
    acc = tl.minimum(acc, inv_scale)

    # store
    out_off = ((pid_n * OC + oc_offs[:, None]) * OH * OW) + sp_offs[None, :]
    mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        N, IC, IH, IW = x.shape
        KH = KW = self.kernel_size
        S = self.stride
        P = self.padding
        OP = self.output_padding
        OH = (IH - 1) * S - 2 * P + KH + OP
        OW = (IW - 1) * S - 2 * P + KW + OP
        OC = self.out_channels

        x = x.contiguous()
        weight = self.conv_transpose.weight.contiguous()  # [IC, OC, KH, KW]
        # combined bias
        combined_bias = (self.conv_transpose.bias.view(OC, 1, 1) + self.bias).view(OC).contiguous()

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_SP = 128
        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_SP))

        inv_scale = 1.0 / self.scaling_factor

        conv_transpose2d_gather_kernel[grid](
            x, weight, combined_bias, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            S, S,
            P, P,
            inv_scale,
            BLOCK_OC=BLOCK_OC,
            BLOCK_SP=BLOCK_SP,
            num_warps=4,
            num_stages=2,
        )

        # final result = clamp(acc,0,1/s) ... but we need (clamp(clamp(x,0,1)*s,0,1))/s
        # clamp(x,0,1) in [0,1]; *s in [0,s]; clamp [0,1]; /s in [0,1/s]
        # which equals min(clamp(x,0,1), 1/s) = min(max(x,0), 1, 1/s) = clamp(x, 0, min(1, 1/s))
        # We applied: clamp(acc,0,1) then min(., inv_scale) -> equals clamp(acc, 0, min(1,inv_scale))
        # That matches the chain only if scaling_factor >= 1 (so inv_scale<=1). For s<1, min(1,1/s)=1.
        # Original: clamp(clamp(x,0,1)*s,0,1)/s. With s<1: clamp(x,0,1)*s <= s <1, so = clamp(x,0,1)*s/s = clamp(x,0,1).
        # Our impl: min(clamp(x,0,1), 1/s) = clamp(x,0,1) since 1/s>1. OK.
        # With s>=1: clamp(clamp(x,0,1)*s,0,1) = min(clamp(x,0,1)*s, 1) -> /s = min(clamp(x,0,1), 1/s). OK.
        return out