import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Direct gather-based ConvTranspose3d with fused add + hardswish epilogue.
# Each program computes a tile of output for one (n, oc_block, spatial_block).
# Output[n, oc, od, oh, ow] = sum_{ic, kd, kh, kw} input[n, ic, id, ih, iw] * weight[ic, oc, kd, kh, kw]
# where id = (od + pad - kd)/stride, valid when (od+pad-kd) % stride == 0 and 0 <= id < D_in

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 256}, num_warps=8, num_stages=2),
    ],
    key=['N', 'IC', 'OC', 'D_in', 'H_in', 'W_in', 'D_out', 'H_out', 'W_out'],
)
@triton.jit
def conv_transpose3d_fused_kernel(
    x_ptr, w_ptr, conv_bias_ptr, add_ptr, out_ptr,
    N, IC, OC,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    HW_out = H_out * W_out
    DHW_out = D_out * HW_out
    DHW_in = D_in * H_in * W_in

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < DHW_out

    od = sp_offs // HW_out
    rem = sp_offs % HW_out
    oh = rem // W_out
    ow = rem % W_out

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    ic_offs = tl.arange(0, BLOCK_IC)
    ic_mask = ic_offs < IC

    n_base = pid_n * (IC * DHW_in)

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    KDHW = KD * KH * KW
    OC_KDHW = OC * KDHW

    for kd in tl.static_range(0, KD):
        id_ = od + PAD - kd
        id_div = id_ // STRIDE
        d_valid = ((id_ % STRIDE) == 0) & (id_div >= 0) & (id_div < D_in)
        for kh in tl.static_range(0, KH):
            ih_ = oh + PAD - kh
            ih_div = ih_ // STRIDE
            h_valid = ((ih_ % STRIDE) == 0) & (ih_div >= 0) & (ih_div < H_in)
            for kw in tl.static_range(0, KW):
                iw_ = ow + PAD - kw
                iw_div = iw_ // STRIDE
                w_valid = ((iw_ % STRIDE) == 0) & (iw_div >= 0) & (iw_div < W_in)
                spatial_valid = d_valid & h_valid & w_valid & sp_mask  # [BLOCK_SP]

                in_spatial_off = id_div * (H_in * W_in) + ih_div * W_in + iw_div  # [BLOCK_SP]

                k_off = (kd * KH + kh) * KW + kw

                # Load input tile: [BLOCK_SP, BLOCK_IC]
                x_off = n_base + ic_offs[None, :] * DHW_in + in_spatial_off[:, None]
                x_m = spatial_valid[:, None] & ic_mask[None, :]
                x_tile = tl.load(x_ptr + x_off, mask=x_m, other=0.0)

                # Load weight tile: [BLOCK_IC, BLOCK_OC]
                w_off = ic_offs[:, None] * OC_KDHW + oc_offs[None, :] * KDHW + k_off
                w_m = ic_mask[:, None] & oc_mask[None, :]
                w_tile = tl.load(w_ptr + w_off, mask=w_m, other=0.0)

                acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    cb = tl.load(conv_bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += cb[None, :]

    out_off = pid_n * (OC * DHW_out) + oc_offs[None, :] * DHW_out + sp_offs[:, None]
    full_mask = sp_mask[:, None] & oc_mask[None, :]

    add_val = tl.load(add_ptr + out_off, mask=full_mask, other=0.0)
    v = acc + add_val

    hs = v * tl.minimum(tl.maximum(v + 3.0, 0.0), 6.0) * (1.0 / 6.0)
    out_val = v * hs

    tl.store(out_ptr + out_off, out_val, mask=full_mask)


def conv_transpose3d_fused(x, weight, conv_bias, add_input,
                            stride, padding, output_padding,
                            D_out, H_out, W_out):
    N, IC, D_in, H_in, W_in = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    assert IC == IC_w

    x = x.contiguous()
    weight = weight.contiguous()
    add_input = add_input.contiguous()

    out = torch.empty((N, OC, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    DHW_out = D_out * H_out * W_out

    # BLOCK_IC chosen to cover full IC=32 in one tile (power of 2, >=16 for tl.dot)
    BLOCK_IC = max(16, triton.next_power_of_2(IC))

    grid = lambda meta: (N, triton.cdiv(OC, meta['BLOCK_OC']), triton.cdiv(DHW_out, meta['BLOCK_SP']))

    conv_transpose3d_fused_kernel[grid](
        x, weight, conv_bias, add_input, out,
        N, IC, OC,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        KD, KH, KW,
        stride, padding,
        BLOCK_IC=BLOCK_IC,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, add_input):
        N, IC, D_in, H_in, W_in = x.shape
        s = self.stride
        p = self.padding
        op = self.output_padding
        k = self.kernel_size
        D_out = (D_in - 1) * s - 2 * p + k + op
        H_out = (H_in - 1) * s - 2 * p + k + op
        W_out = (W_in - 1) * s - 2 * p + k + op

        return conv_transpose3d_fused(
            x, self.conv_transpose.weight, self.conv_transpose.bias, add_input,
            s, p, op, D_out, H_out, W_out,
        )