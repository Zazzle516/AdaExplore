import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 256}, num_warps=8, num_stages=2),
    ],
    key=['C_IN', 'C_OUT', 'H_OUT', 'W_OUT'],
)
@triton.jit
def conv2d_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_IN, H_IN, W_IN,
    C_OUT, H_OUT, W_OUT,
    KH: tl.constexpr, KW: tl.constexpr,
    SUB: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    HW_OUT = H_OUT * W_OUT
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_sp = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    mask_oc = offs_oc < C_OUT
    mask_sp = offs_sp < HW_OUT

    oh = offs_sp // W_OUT
    ow = offs_sp % W_OUT

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # input base pointer for this batch
    x_n_base = pid_n * C_IN * H_IN * W_IN
    HW_IN = H_IN * W_IN
    KHW = KH * KW
    C_IN_KHW = C_IN * KHW

    # base spatial offset in input
    sp_base = oh * W_IN + ow  # [BLOCK_SP]

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh
            iw = ow + kw
            spatial_mask = mask_sp & (ih < H_IN) & (iw < W_IN)
            x_sp_offset = sp_base + kh * W_IN + kw  # [BLOCK_SP]
            w_kk_offset = kh * KW + kw  # scalar
            for ic in range(0, C_IN):
                x_offset = x_n_base + ic * HW_IN + x_sp_offset
                x_vals = tl.load(x_ptr + x_offset, mask=spatial_mask, other=0.0)

                w_offset = offs_oc * C_IN_KHW + ic * KHW + w_kk_offset
                w_vals = tl.load(w_ptr + w_offset, mask=mask_oc, other=0.0)

                acc += w_vals[:, None] * x_vals[None, :]

    # bias
    b_vals = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + b_vals[:, None]

    # subtract
    acc = acc - SUB

    # mish: x * tanh(softplus(x))
    sp = tl.log(1.0 + tl.exp(acc))
    # tanh via exp
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    out = acc * th

    # store
    out_n_base = pid_n * C_OUT * HW_OUT
    out_offset = out_n_base + offs_oc[:, None] * HW_OUT + offs_sp[None, :]
    out_mask = mask_oc[:, None] & mask_sp[None, :]
    tl.store(out_ptr + out_offset, out, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, C_IN, H_IN, W_IN = x.shape
        KH = KW = self.kernel_size
        H_OUT = H_IN - KH + 1
        W_OUT = W_IN - KW + 1
        C_OUT = self.out_channels

        out = torch.empty((N, C_OUT, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

        SUB = float(self.subtract_value_1 + self.subtract_value_2)

        grid = lambda meta: (N, triton.cdiv(C_OUT, meta['BLOCK_OC']), triton.cdiv(H_OUT * W_OUT, meta['BLOCK_SP']))

        conv2d_mish_kernel[grid](
            x, self.conv.weight, self.conv.bias, out,
            N, C_IN, H_IN, W_IN,
            C_OUT, H_OUT, W_OUT,
            KH, KW,
            SUB,
        )
        return out