import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _conv_transpose2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_SP: tl.constexpr,
):
    pid_sp = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_n = tl.program_id(2)

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    sp_mask = sp_offs < (OH * OW)
    oc_mask = oc_offs < OC

    oh = sp_offs // OW
    ow = sp_offs % OW

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # For ConvTranspose2d: out[n, oc, oh, ow] = sum_{ic, kh, kw} x[n, ic, ih, iw] * w[ic, oc, kh, kw]
    # where ih*stride - pad + kh = oh, so ih = (oh + pad - kh) / stride
    # Pre-transposed weight: w_t[oc, ic, kh, kw]
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih_num = oh + PAD - kh
            iw_num = ow + PAD - kw
            ih = ih_num // STRIDE
            iw = iw_num // STRIDE
            valid = (ih_num % STRIDE == 0) & (iw_num % STRIDE == 0) & \
                    (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW) & sp_mask

            for ic in range(0, IC):
                x_off = pid_n * IC * IH * IW + ic * IH * IW + ih * IW + iw
                x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)  # [BLOCK_SP]

                w_off = oc_offs * IC * KH * KW + ic * KH * KW + kh * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                acc += x_val[:, None] * w_val[None, :]

    # bias from conv_transpose + subtract self.bias + tanh
    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + b_val[None, :]
    e2x = tl.exp(2.0 * acc)
    out = (e2x - 1.0) / (e2x + 1.0)

    # Store: out[n, oc, oh, ow], layout NCHW
    out_off = pid_n * OC * OH * OW + oc_offs[None, :] * OH * OW + sp_offs[:, None]
    mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        # Use the same init as nn.ConvTranspose2d
        conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                  stride=stride, padding=padding, output_padding=output_padding)
        # weight shape: [IC, OC, KH, KW]
        self.conv_weight = nn.Parameter(conv.weight.detach().clone())
        self.conv_bias = nn.Parameter(conv.bias.detach().clone())
        self.bias = nn.Parameter(torch.randn(bias_shape))

        # Pre-transposed weight: [OC, IC, KH, KW]
        self._weight_t = None

    def _get_weight_t(self):
        # Permute [IC, OC, KH, KW] -> [OC, IC, KH, KW]
        return self.conv_weight.permute(1, 0, 2, 3).contiguous()

    def forward(self, x):
        x = x.contiguous()
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        stride = self.stride
        pad = self.padding
        outpad = self.output_padding

        OH = (IH - 1) * stride - 2 * pad + KH + outpad
        OW = (IW - 1) * stride - 2 * pad + KW + outpad

        # Effective conv bias: conv_bias - self.bias (broadcast over OC)
        # self.bias shape (OC,1,1) -> view as (OC,)
        eff_bias = self.conv_bias - self.bias.view(-1)
        eff_bias = eff_bias.contiguous()

        weight_t = self._get_weight_t()

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_SP = 64

        grid = (
            (OH * OW + BLOCK_SP - 1) // BLOCK_SP,
            (OC + BLOCK_OC - 1) // BLOCK_OC,
            N,
        )

        _conv_transpose2d_kernel[grid](
            x, weight_t, eff_bias, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH=KH, KW=KW,
            STRIDE=stride, PAD=pad,
            BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
            num_warps=4, num_stages=2,
        )

        return out