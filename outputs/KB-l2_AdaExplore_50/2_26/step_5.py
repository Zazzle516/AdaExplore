import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def convt3d_fused_kernel(
    x_ptr, w_ptr, add_ptr, out_ptr,
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_SP: tl.constexpr,
    IC_C: tl.constexpr,
):
    # program ids
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    OHW = OH * OW
    OSPATIAL = OD * OH * OW
    sp_mask = sp_offs < OSPATIAL

    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    # accumulator [BLOCK_SP, BLOCK_OC]
    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # For each kernel tap (kd, kh, kw), determine which input element contributes
    # Output index: od = id*STRIDE - PAD + kd  =>  id = (od + PAD - kd) / STRIDE
    # Must be divisible by STRIDE and within [0, ID)

    for kd in tl.static_range(KD):
        id_num = od + PAD - kd
        id_val = id_num // STRIDE
        id_ok = ((id_num - id_val * STRIDE) == 0) & (id_val >= 0) & (id_val < ID)
        for kh in tl.static_range(KH):
            ih_num = oh + PAD - kh
            ih_val = ih_num // STRIDE
            ih_ok = ((ih_num - ih_val * STRIDE) == 0) & (ih_val >= 0) & (ih_val < IH)
            for kw in tl.static_range(KW):
                iw_num = ow + PAD - kw
                iw_val = iw_num // STRIDE
                iw_ok = ((iw_num - iw_val * STRIDE) == 0) & (iw_val >= 0) & (iw_val < IW)

                valid = id_ok & ih_ok & iw_ok & sp_mask  # [BLOCK_SP]

                # input pointer base for this tap (varies per spatial)
                # x[pid_n, ic, id_val, ih_val, iw_val]
                # We iterate ic in chunks
                spatial_in = id_val * (IH * IW) + ih_val * IW + iw_val  # [BLOCK_SP]
                x_base = pid_n * (IC * ID * IH * IW) + spatial_in  # [BLOCK_SP]
                # weight: w[ic, oc, kd, kh, kw], stride: (OC*KD*KH*KW, KD*KH*KW, KH*KW, KW, 1)
                w_base_kk = (kd * KH * KW) + (kh * KW) + kw  # scalar
                # for each ic chunk
                for ic_start in range(0, IC, IC_C):
                    ic_offs = ic_start + tl.arange(0, IC_C)
                    ic_mask = ic_offs < IC

                    # Load x: shape [BLOCK_SP, IC_C]
                    x_ptrs = x_ptr + x_base[:, None] + ic_offs[None, :] * (ID * IH * IW)
                    x_load_mask = valid[:, None] & ic_mask[None, :]
                    x_vals = tl.load(x_ptrs, mask=x_load_mask, other=0.0)

                    # Load w: shape [IC_C, BLOCK_OC]
                    # w_ptr + ic*OC*KD*KH*KW + oc*KD*KH*KW + w_base_kk
                    w_ptrs = w_ptr + ic_offs[:, None] * (OC * KD * KH * KW) + oc_offs[None, :] * (KD * KH * KW) + w_base_kk
                    w_load_mask = ic_mask[:, None] & oc_mask[None, :]
                    w_vals = tl.load(w_ptrs, mask=w_load_mask, other=0.0)

                    acc += tl.dot(x_vals, w_vals)

    # Now load add_input and apply hardswish-mul fusion
    # out[pid_n, oc, od, oh, ow]
    out_base = pid_n * (OC * OSPATIAL) + oc_offs[None, :] * OSPATIAL + sp_offs[:, None]
    out_mask = sp_mask[:, None] & oc_mask[None, :]

    add_vals = tl.load(add_ptr + out_base, mask=out_mask, other=0.0)

    v = acc + add_vals
    hs_inner = v + 3.0
    hs_inner = tl.maximum(hs_inner, 0.0)
    hs_inner = tl.minimum(hs_inner, 6.0)
    hs = v * hs_inner * (1.0 / 6.0)
    out_v = v * hs

    tl.store(out_ptr + out_base, out_v, mask=out_mask)


@triton.jit
def add_bias_hardswish_to_out_kernel(
    conv_ptr, add_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    c = tl.load(conv_ptr + offs, mask=mask, other=0.0)
    a = tl.load(add_ptr + offs, mask=mask, other=0.0)
    v = c + a
    hs_inner = v + 3.0
    hs_inner = tl.maximum(hs_inner, 0.0)
    hs_inner = tl.minimum(hs_inner, 6.0)
    hs = v * hs_inner * (1.0 / 6.0)
    tl.store(out_ptr + offs, v * hs, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x, add_input):
        x = x.contiguous()
        add_input = add_input.contiguous()

        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        STRIDE = self.stride
        PAD = self.padding
        OUTPAD = self.output_padding

        OD = (ID - 1) * STRIDE - 2 * PAD + KD + OUTPAD
        OH = (IH - 1) * STRIDE - 2 * PAD + KH + OUTPAD
        OW = (IW - 1) * STRIDE - 2 * PAD + KW + OUTPAD

        # Get conv weight: (IC, OC, KD, KH, KW)
        w = self.conv_transpose.weight.contiguous()
        conv_bias = self.conv_transpose.bias  # (OC,)

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        # If conv has bias, we need to add it. We'll incorporate it by adding to add_input pre-pass.
        # Simpler: pre-add conv_bias to add_input (broadcast) into a working tensor.
        if conv_bias is not None:
            add_eff = add_input + conv_bias.view(1, OC, 1, 1, 1)
        else:
            add_eff = add_input
        add_eff = add_eff.contiguous()

        BLOCK_OC = 32
        BLOCK_SP = 64
        IC_C = 32  # IC is 32

        OSPATIAL = OD * OH * OW
        grid = (
            N,
            triton.cdiv(OC, BLOCK_OC),
            triton.cdiv(OSPATIAL, BLOCK_SP),
        )

        convt3d_fused_kernel[grid](
            x, w, add_eff, out,
            N, IC, OC,
            ID, IH, IW,
            OD, OH, OW,
            KD, KH, KW,
            STRIDE, PAD,
            BLOCK_OC, BLOCK_SP,
            IC_C,
            num_warps=4, num_stages=2,
        )

        return out