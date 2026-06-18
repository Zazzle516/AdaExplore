import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_IC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_IC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 256, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 256, 'BLOCK_IC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 128, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128, 'BLOCK_IC': 64}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'IC', 'H_out', 'W_out'],
)
@triton.jit
def convtranspose2d_gather_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H_in, W_in,
    OC, H_out, W_out,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    ADD_VALUE: tl.constexpr, SCALE: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_SP: tl.constexpr, BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    ic_inner = tl.arange(0, BLOCK_IC)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (H_out * W_out)

    h_out = sp_offs // W_out
    w_out = sp_offs % W_out

    # accumulator [BLOCK_SP, BLOCK_OC]
    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    HW_in = H_in * W_in
    OC_KH_KW = OC * KH * KW

    for kh in tl.static_range(0, KH):
        h_num = h_out + PAD - kh  # [BLOCK_SP]
        h_in = h_num // STRIDE
        h_valid = (h_num >= 0) & ((h_num % STRIDE) == 0) & (h_in < H_in)
        for kw in tl.static_range(0, KW):
            w_num = w_out + PAD - kw
            w_in = w_num // STRIDE
            w_valid = (w_num >= 0) & ((w_num % STRIDE) == 0) & (w_in < W_in)
            valid = h_valid & w_valid & sp_mask  # [BLOCK_SP]

            x_base_sp = pid_n * IC * HW_in + h_in * W_in + w_in  # [BLOCK_SP]
            w_base_oc = oc_offs * KH * KW + kh * KW + kw  # [BLOCK_OC]

            for ic_start in range(0, IC, BLOCK_IC):
                ic_offs = ic_start + ic_inner  # [BLOCK_IC]
                ic_mask = ic_offs < IC

                # x tile [BLOCK_SP, BLOCK_IC]
                x_addr = x_base_sp[:, None] + ic_offs[None, :] * HW_in
                x_mask = valid[:, None] & ic_mask[None, :]
                x_tile = tl.load(x_ptr + x_addr, mask=x_mask, other=0.0)

                # w tile [BLOCK_IC, BLOCK_OC]
                w_addr = ic_offs[:, None] * OC_KH_KW + w_base_oc[None, :]
                w_mask = ic_mask[:, None] & oc_mask[None, :]
                w_tile = tl.load(w_ptr + w_addr, mask=w_mask, other=0.0)

                acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    # add bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc = acc + bias[None, :]

    # Mish: x * tanh(softplus(x))
    sp = tl.log(1.0 + tl.exp(acc))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    y = acc * th
    y = y + ADD_VALUE
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)
    y = y * SCALE

    # store: out[n, oc, h_out, w_out]
    out_offs = (pid_n * OC * H_out * W_out
                + oc_offs[None, :] * H_out * W_out
                + sp_offs[:, None])
    store_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_offs, y, mask=store_mask)


def convtranspose2d_fused(x, weight, bias, stride, padding, output_padding, add_value, scale):
    N, IC, H_in, W_in = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w

    H_out = (H_in - 1) * stride - 2 * padding + KH + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + KW + output_padding

    out = torch.empty((N, OC, H_out, W_out), device=x.device, dtype=x.dtype)

    grid = lambda meta: (N, triton.cdiv(OC, meta['BLOCK_OC']), triton.cdiv(H_out * W_out, meta['BLOCK_SP']))

    convtranspose2d_gather_kernel[grid](
        x, weight, bias, out,
        N, IC, H_in, W_in,
        OC, H_out, W_out,
        KH=KH, KW=KW,
        STRIDE=stride, PAD=padding,
        ADD_VALUE=float(add_value), SCALE=float(scale),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()
        b = self.conv_transpose.bias.contiguous()
        return convtranspose2d_fused(
            x, w, b,
            self.stride, self.padding, self.output_padding,
            self.add_value, self.scale,
        )