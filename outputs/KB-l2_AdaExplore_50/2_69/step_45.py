import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 128}, num_warps=4, num_stages=2),
    ],
    key=['N', 'OC', 'IC', 'H', 'W', 'KH', 'KW'],
)
@triton.jit
def conv_hardswish_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wh, stride_ww,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_N: tl.constexpr, BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)  # over N*OH*OW tile
    pid_oc = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    NHW = N * OH * OW
    mask_n = offs_n < NHW
    mask_oc = offs_oc < OC

    # decode n, oh, ow from offs_n
    n_idx = offs_n // (OH * OW)
    rem = offs_n % (OH * OW)
    oh_idx = rem // OW
    ow_idx = rem % OW

    # Hoist constant index math out of the inner loops
    x_base = n_idx * stride_xn + oh_idx * stride_xh + ow_idx * stride_xw  # [BLOCK_N]
    w_base = offs_oc * stride_wo  # [BLOCK_OC]

    ic_offs = tl.arange(0, BLOCK_IC)
    ic_mask = ic_offs < IC

    acc = tl.zeros((BLOCK_N, BLOCK_OC), dtype=tl.float32)

    # For each (kh, kw), do a tl.dot with K = BLOCK_IC (padded from IC)
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            x_spatial = x_base + kh * stride_xh + kw * stride_xw  # [BLOCK_N]
            w_spatial = w_base + kh * stride_wh + kw * stride_ww  # [BLOCK_OC]

            # x: [BLOCK_N, BLOCK_IC]
            x_off = x_spatial[:, None] + (ic_offs * stride_xc)[None, :]
            x_msk = mask_n[:, None] & ic_mask[None, :]
            x_vals = tl.load(x_ptr + x_off, mask=x_msk, other=0.0)

            # w: [BLOCK_IC, BLOCK_OC]
            w_off = (ic_offs * stride_wi)[:, None] + w_spatial[None, :]
            w_msk = ic_mask[:, None] & mask_oc[None, :]
            w_vals = tl.load(w_ptr + w_off, mask=w_msk, other=0.0)

            acc += tl.dot(x_vals, w_vals, allow_tf32=False)

    # bias
    b_vals = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + b_vals[None, :]

    # hardswish: x * relu6(x+3) / 6, then relu
    # combined: if x <= 0 -> 0; if x >= 3 -> x; else x*(x+3)/6
    # After relu, anything <=0 becomes 0 anyway.
    x = acc
    hs = x * tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0) / 6.0
    out = tl.maximum(hs, 0.0)

    # store
    out_off = (n_idx * stride_on)[:, None] + (offs_oc * stride_oc)[None, :] + (oh_idx * stride_oh)[:, None] + (ow_idx * stride_ow)[:, None]
    mask = mask_n[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_off, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.cuda().contiguous()
        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()

        N, IC, H, W = x.shape
        OC, _, KH, KW = weight.shape
        OH = H - KH + 1
        OW = W - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        grid = lambda meta: (
            triton.cdiv(N * OH * OW, meta['BLOCK_N']),
            triton.cdiv(OC, meta['BLOCK_OC']),
        )

        BLOCK_IC = 16 if IC <= 16 else triton.next_power_of_2(IC)

        conv_hardswish_relu_kernel[grid](
            x, weight, bias, out,
            N, IC, H, W,
            OC, OH, OW,
            KH, KW,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            BLOCK_IC=BLOCK_IC,
        )
        return out