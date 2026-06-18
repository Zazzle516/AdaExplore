import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 128}, num_warps=8, num_stages=2),
    ],
    key=['N_OUT', 'OC', 'KH', 'KW', 'IC'],
)
@triton.jit
def conv_hswish_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, IH, IW,
    OC, OH, OW,
    IC: tl.constexpr,
    IC_PAD: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    N_OUT,  # B * OH * OW
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wh, stride_ww,
    stride_ob, stride_oc, stride_oh, stride_ow,
    BLOCK_N: tl.constexpr, BLOCK_OC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_ic = tl.arange(0, IC_PAD)

    mask_n = offs_n < N_OUT
    mask_oc = offs_oc < OC
    mask_ic = offs_ic < IC

    # decode N axis -> (b, oh, ow)
    ow = offs_n % OW
    tmp = offs_n // OW
    oh = tmp % OH
    b = tmp // OH

    acc = tl.zeros((BLOCK_N, BLOCK_OC), dtype=tl.float32)

    # Hoist base addresses
    x_base_n = b * stride_xb + oh * stride_xh + ow * stride_xw  # [BLOCK_N]
    x_ic_off = offs_ic * stride_xc  # [IC_PAD]
    w_oc_off = offs_oc * stride_wo  # [BLOCK_OC]
    w_ic_off = offs_ic * stride_wi  # [IC_PAD]

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            # x tile: [BLOCK_N, IC_PAD]
            x_off = (x_base_n + kh * stride_xh + kw * stride_xw)[:, None] + x_ic_off[None, :]
            x_mask = mask_n[:, None] & mask_ic[None, :]
            x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

            # w tile: [IC_PAD, BLOCK_OC]
            w_off = w_ic_off[:, None] + w_oc_off[None, :] + (kh * stride_wh + kw * stride_ww)
            w_mask = mask_ic[:, None] & mask_oc[None, :]
            w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

            acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + bias[None, :]

    # relu(hardswish(x)) = relu(x * relu6(x+3)/6)
    relu6 = tl.minimum(tl.maximum(acc + 3.0, 0.0), 6.0)
    hs = acc * relu6 * (1.0 / 6.0)
    out_val = tl.maximum(hs, 0.0)

    out_off = (b * stride_ob + oh * stride_oh + ow * stride_ow)[:, None] + offs_oc[None, :] * stride_oc
    mask = mask_n[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_off, out_val, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.cuda().contiguous()
        w = self.conv.weight.cuda().contiguous()
        b = self.conv.bias.cuda().contiguous()

        B, IC, IH, IW = x.shape
        OC, _, KH, KW = w.shape
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((B, OC, OH, OW), device=x.device, dtype=x.dtype)

        N_OUT = B * OH * OW

        grid = lambda meta: (
            triton.cdiv(N_OUT, meta['BLOCK_N']),
            triton.cdiv(OC, meta['BLOCK_OC']),
        )

        # pad IC to next power of 2, min 16 for tl.dot
        IC_PAD = 16
        while IC_PAD < IC:
            IC_PAD *= 2

        conv_hswish_relu_kernel[grid](
            x, w, b, out,
            B, IH, IW,
            OC, OH, OW,
            IC, IC_PAD, KH, KW,
            N_OUT,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w.stride(0), w.stride(1), w.stride(2), w.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        )
        return out