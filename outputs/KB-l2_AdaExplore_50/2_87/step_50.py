import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 512, 'BLOCK_OC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_OC': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 512, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 128}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'KH', 'KW', 'OH', 'OW'],
)
@triton.jit
def conv2d_sub_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    SUB: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_OC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_b = tl.program_id(2)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # spatial pos within OH*OW
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    n_mask = n_offs < (OH * OW)
    oc_mask = oc_offs < OC

    oh = n_offs // OW
    ow = n_offs % OW

    # Base offset into x for this batch's spatial tile (without kh, kw, ic)
    x_batch_base = pid_b * IC * IH * IW
    x_spatial_base = oh * IW + ow  # [BLOCK_N], base spatial offset within (IH,IW)

    # Base offset into w for these output channels (without kh, kw, ic)
    w_oc_base = oc_offs * (IC * KH * KW)  # [BLOCK_OC]

    acc = tl.zeros((BLOCK_N, BLOCK_OC), dtype=tl.float32)

    # Loop over IC, KH, KW (unroll KH/KW)
    for ic in range(0, IC):
        x_ic_base = x_batch_base + ic * IH * IW
        w_ic_base = w_oc_base + ic * KH * KW
        for kh in tl.static_range(0, KH):
            x_kh_base = x_ic_base + kh * IW
            w_kh_base = w_ic_base + kh * KW
            for kw in tl.static_range(0, KW):
                x_offset = x_kh_base + x_spatial_base + kw
                x_vals = tl.load(x_ptr + x_offset, mask=n_mask, other=0.0)  # [BLOCK_N]

                w_offset = w_kh_base + kw
                w_vals = tl.load(w_ptr + w_offset, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                acc += x_vals[:, None] * w_vals[None, :]

    # bias
    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + b_vals[None, :] - SUB

    # mish: x * tanh(softplus(x)) = x * tanh(log(1+exp(x)))
    sp = tl.log(1.0 + tl.exp(acc))
    # tanh via exp
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    out = acc * th

    # store: out[b, oc, oh, ow]
    out_offset = pid_b * OC * OH * OW + oc_offs[None, :] * OH * OW + n_offs[:, None]
    out_mask = n_mask[:, None] & oc_mask[None, :]
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
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        sub = float(self.subtract_value_1 + self.subtract_value_2)

        grid = lambda meta: (
            triton.cdiv(OH * OW, meta['BLOCK_N']),
            triton.cdiv(OC, meta['BLOCK_OC']),
            N,
        )

        conv2d_sub_mish_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            sub,
        )
        return out