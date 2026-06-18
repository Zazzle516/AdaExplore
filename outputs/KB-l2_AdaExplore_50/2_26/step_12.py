import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 64, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_IC': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 256, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 256, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 256, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'OC', 'OD', 'OH', 'OW'],
)
@triton.jit
def conv_transpose3d_fused_kernel(
    x_ptr, w_ptr, conv_bias_ptr, add_input_ptr, out_ptr,
    N, IC, OC, ID, IH, IW, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_SP: tl.constexpr, BLOCK_IC: tl.constexpr,
):
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

    ic_arange = tl.arange(0, BLOCK_IC)

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    KDHW = KD * KH * KW
    OC_KDHW = OC * KDHW
    ID_IH_IW = ID * IH * IW
    IH_IW = IH * IW

    x_batch_base = pid_n * (IC * ID_IH_IW)

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
                in_spatial_idx = id_val * IH_IW + ih_val * IW + iw_val
                in_spatial_idx_safe = tl.where(spatial_valid, in_spatial_idx, 0)

                kidx = kd * (KH * KW) + kh * KW + kw

                for ic_start in range(0, IC, BLOCK_IC):
                    ic_offs = ic_start + ic_arange  # [BLOCK_IC]
                    ic_mask = ic_offs < IC

                    # x: [BLOCK_IC, BLOCK_SP]
                    x_offs = x_batch_base + ic_offs[:, None] * ID_IH_IW + in_spatial_idx_safe[None, :]
                    x_load_mask = ic_mask[:, None] & spatial_valid[None, :]
                    x_vals = tl.load(x_ptr + x_offs, mask=x_load_mask, other=0.0)

                    # w: [BLOCK_OC, BLOCK_IC] from weight[ic, oc, kd, kh, kw]
                    w_offs = ic_offs[None, :] * OC_KDHW + oc_offs[:, None] * KDHW + kidx
                    w_load_mask = oc_mask[:, None] & ic_mask[None, :]
                    w_vals = tl.load(w_ptr + w_offs, mask=w_load_mask, other=0.0)

                    acc += tl.dot(w_vals, x_vals, allow_tf32=True)

    # Add conv bias
    cbias = tl.load(conv_bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += cbias[:, None]

    # Add the add_input tensor
    add_offs = pid_n * (OC * DHW) + oc_offs[:, None] * DHW + sp_offs[None, :]
    add_mask = oc_mask[:, None] & sp_mask[None, :]
    add_vals = tl.load(add_input_ptr + add_offs, mask=add_mask, other=0.0)
    acc += add_vals

    # HardSwish: x * hardswish(x) = x * x * relu6(x+3)/6
    hs_inner = acc + 3.0
    hs_inner = tl.maximum(hs_inner, 0.0)
    hs_inner = tl.minimum(hs_inner, 6.0)
    hardswish_val = acc * hs_inner * (1.0 / 6.0)
    result = acc * hardswish_val

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

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_OC']),
            triton.cdiv(OD * OH * OW, meta['BLOCK_SP']),
        )

        conv_transpose3d_fused_kernel[grid](
            x, weight, conv_bias, add_input, out,
            N, IC, OC, ID, IH, IW, OD, OH, OW,
            KD, KH, KW, S, P,
        )
        return out