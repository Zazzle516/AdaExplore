import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64,  'BLOCK_SP': 64,  'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64,  'BLOCK_SP': 64,  'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64,  'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64,  'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64,  'BLOCK_SP': 128, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64,  'BLOCK_SP': 128, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 128, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 128, 'BLOCK_IC': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 32,  'BLOCK_SP': 128, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64,  'BLOCK_SP': 256, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'HW', 'IC'],
)
@triton.jit
def conv_transpose_gather_kernel(
    x_ptr,      # [N, IC, H_in, W_in]
    w_ptr,      # [IC, OC, KH, KW]
    b_ptr,      # [OC]
    out_ptr,    # [N, OC, H_out, W_out]
    N, IC, OC,
    H_in, W_in,
    H_out, W_out, HW,
    add_value: tl.constexpr,
    multiply_value: tl.constexpr,
    STRIDE: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    sp_off = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_off < HW

    h_out = sp_off // W_out
    w_out = sp_off % W_out

    oc_off = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_off < OC

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    ic_range = tl.arange(0, BLOCK_IC)

    # Outer loop over IC tiles; inner static loops over (kh, kw) reuse x_base/w pointers
    HW_in = H_in * W_in
    x_n_base = pid_n * (IC * HW_in)

    for ic_start in range(0, IC, BLOCK_IC):
        ic_off = ic_start + ic_range  # [BLOCK_IC]
        ic_mask = ic_off < IC

        for kh in tl.static_range(0, KH):
            h_in_num = h_out - kh
            h_valid = (h_in_num >= 0) & ((h_in_num % STRIDE) == 0)
            h_in = h_in_num // STRIDE
            h_valid = h_valid & (h_in < H_in)

            for kw in tl.static_range(0, KW):
                w_in_num = w_out - kw
                w_valid = (w_in_num >= 0) & ((w_in_num % STRIDE) == 0)
                w_in = w_in_num // STRIDE
                w_valid = w_valid & (w_in < W_in)

                sp_valid = h_valid & w_valid & sp_mask  # [BLOCK_SP]

                # x_tile: [BLOCK_SP, BLOCK_IC]
                x_sp_off = h_in * W_in + w_in  # [BLOCK_SP]
                x_ptrs = x_ptr + x_n_base + ic_off[None, :] * HW_in + x_sp_off[:, None]
                x_mask = sp_valid[:, None] & ic_mask[None, :]
                x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

                # w_tile: [BLOCK_IC, BLOCK_OC]
                w_ptrs = (w_ptr
                          + ic_off[:, None] * (OC * KH * KW)
                          + oc_off[None, :] * (KH * KW)
                          + (kh * KW + kw))
                w_mask = ic_mask[:, None] & oc_mask[None, :]
                w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

                acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    # bias
    b_vals = tl.load(b_ptr + oc_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc = acc + b_vals[None, :]

    # epilogue: + add_value, min(., 0), GELU, * multiply_value
    acc = acc + add_value
    acc = tl.minimum(acc, 0.0)
    inv_sqrt2 = 0.7071067811865475
    g = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))
    g = g * multiply_value

    # store: out[n, oc, h_out, w_out]
    out_base = pid_n * (OC * HW)
    out_off = out_base + oc_off[None, :] * HW + sp_off[:, None]
    store_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, g, mask=store_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = float(add_value)
        self.multiply_value = float(multiply_value)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = x.contiguous()
        N, IC, H_in, W_in = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        S = self.stride
        H_out = (H_in - 1) * S + KH
        W_out = (W_in - 1) * S + KW

        weight = self.conv_transpose.weight.contiguous()  # [IC, OC, KH, KW]
        bias = self.conv_transpose.bias.contiguous()      # [OC]

        out = torch.empty((N, OC, H_out, W_out), device=x.device, dtype=x.dtype)

        HW = H_out * W_out

        grid = lambda META: (
            N,
            triton.cdiv(OC, META['BLOCK_OC']),
            triton.cdiv(HW, META['BLOCK_SP']),
        )

        conv_transpose_gather_kernel[grid](
            x, weight, bias, out,
            N, IC, OC,
            H_in, W_in,
            H_out, W_out, HW,
            add_value=self.add_value,
            multiply_value=self.multiply_value,
            STRIDE=S,
            KH=KH,
            KW=KW,
        )
        return out