import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_fused_kernel(
    x_ptr, w_ptr, conv_bias_ptr, add_bias_ptr, add_input_ptr, out_ptr,
    N, IC, OC, ID, IH, IW, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_SP: tl.constexpr,
):
    # program: (n, oc_block, spatial_block)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    DHW = OD * OH * OW
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < DHW

    ow = sp_offs % OW
    oh = (sp_offs // OW) % OH
    od = sp_offs // (OW * OH)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # For ConvTranspose3d:
    # output[od, oh, ow] = sum over (ic, kd, kh, kw) of
    #   input[ic, id, ih, iw] * weight[ic, oc, kd, kh, kw]
    # where: od = id*STRIDE - PAD + kd  =>  id = (od + PAD - kd) / STRIDE
    # and (od + PAD - kd) must be divisible by STRIDE and id in [0, ID)

    for kd in tl.static_range(0, KD):
        id_num = od + PAD - kd
        id_val = id_num // STRIDE
        id_valid = (id_num >= 0) & ((id_num % STRIDE) == 0) & (id_val >= 0) & (id_val < ID)
        for kh in tl.static_range(0, KH):
            ih_num = oh + PAD - kh
            ih_val = ih_num // STRIDE
            ih_valid = (ih_num >= 0) & ((ih_num % STRIDE) == 0) & (ih_val >= 0) & (ih_val < IH)
            for kw in tl.static_range(0, KW):
                iw_num = ow + PAD - kw
                iw_val = iw_num // STRIDE
                iw_valid = (iw_num >= 0) & ((iw_num % STRIDE) == 0) & (iw_val >= 0) & (iw_val < IW)

                spatial_valid = id_valid & ih_valid & iw_valid & sp_mask
                # input idx within batch: ic*ID*IH*IW + id*IH*IW + ih*IW + iw
                in_spatial_idx = id_val * (IH * IW) + ih_val * IW + iw_val
                # safe indices for masked load
                in_spatial_idx_safe = tl.where(spatial_valid, in_spatial_idx, 0)

                # weight shape: (IC, OC, KD, KH, KW), index: ic*OC*KD*KH*KW + oc*KD*KH*KW + kd*KH*KW + kh*KW + kw
                kidx = kd * (KH * KW) + kh * KW + kw

                # Accumulate over IC
                for ic in range(0, IC):
                    # load input: x[n, ic, id, ih, iw]
                    x_base = pid_n * (IC * ID * IH * IW) + ic * (ID * IH * IW)
                    x_offs = x_base + in_spatial_idx_safe  # [BLOCK_SP]
                    x_vals = tl.load(x_ptr + x_offs, mask=spatial_valid, other=0.0)  # [BLOCK_SP]

                    # load weight: w[ic, oc, kd, kh, kw] for all oc in block
                    w_base = ic * (OC * KD * KH * KW) + kidx
                    w_offs = w_base + oc_offs * (KD * KH * KW)  # [BLOCK_OC]
                    w_vals = tl.load(w_ptr + w_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    # outer product
                    acc += w_vals[:, None] * x_vals[None, :]

    # Add conv bias
    cbias = tl.load(conv_bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += cbias[:, None]

    # Add the add_input tensor: shape (N, OC, OD, OH, OW)
    add_offs = pid_n * (OC * DHW) + oc_offs[:, None] * DHW + sp_offs[None, :]
    add_mask = oc_mask[:, None] & sp_mask[None, :]
    add_vals = tl.load(add_input_ptr + add_offs, mask=add_mask, other=0.0)
    acc += add_vals

    # Add the learned bias parameter (shape (OC,1,1,1,1))
    abias = tl.load(add_bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    # Wait - in the model, bias is defined but NOT used in forward!
    # Looking again at Model.forward: it never uses self.bias. So skip it.
    # (Leaving load but not adding)

    # HardSwish: x * hardswish(x) = x * x * relu6(x+3)/6
    # hardswish(x) = x * F.relu6(x+3) / 6
    three = 3.0
    six = 6.0
    hs_inner = acc + three
    hs_inner = tl.maximum(hs_inner, 0.0)
    hs_inner = tl.minimum(hs_inner, six)
    hardswish_val = acc * hs_inner / six
    result = acc * hardswish_val

    # Store: out[n, oc, od, oh, ow]
    out_offs = pid_n * (OC * DHW) + oc_offs[:, None] * DHW + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_offs, result, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        # Match nn.ConvTranspose3d initialization
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x, add_input):
        x = x.contiguous()
        add_input = add_input.contiguous()
        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, KD, KH, KW)
        conv_bias = self.conv_transpose.bias.contiguous()  # (OC,)
        add_bias = self.bias.view(-1).contiguous()  # (OC,) - unused but passed

        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        S = self.stride
        P = self.padding
        OP = self.output_padding

        OD = (ID - 1) * S - 2 * P + KD + OP
        OH = (IH - 1) * S - 2 * P + KH + OP
        OW = (IW - 1) * S - 2 * P + KW + OP

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_SP = 128

        grid = (
            N,
            triton.cdiv(OC, BLOCK_OC),
            triton.cdiv(OD * OH * OW, BLOCK_SP),
        )

        conv_transpose3d_fused_kernel[grid](
            x, weight, conv_bias, add_bias, add_input, out,
            N, IC, OC, ID, IH, IW, OD, OH, OW,
            KD, KH, KW, S, P,
            BLOCK_OC, BLOCK_SP,
            num_warps=4, num_stages=2,
        )
        return out