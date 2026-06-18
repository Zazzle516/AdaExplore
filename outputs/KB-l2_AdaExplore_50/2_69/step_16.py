import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ],
    key=['N_OUT', 'OC', 'IC_PADDED', 'KH', 'KW'],
)
@triton.jit
def conv_hswish_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, IC_PADDED: tl.constexpr, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    N_OUT,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wk, stride_wo,
    stride_ob, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < N_OUT
    mask_n = offs_n < OC

    ow = offs_m % OW
    tmp = offs_m // OW
    oh = tmp % OH
    b = tmp // OH

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over (kh, kw); inner K = IC_PADDED (static)
    ic_range = tl.arange(0, IC_PADDED)  # [IC_PADDED]

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh  # [BLOCK_M]
            iw = ow + kw  # [BLOCK_M]
            # x: [BLOCK_M, IC_PADDED]
            x_off = (b * stride_xb + ih * stride_xh + iw * stride_xw)[:, None] + ic_range[None, :] * stride_xc
            x_vals = tl.load(x_ptr + x_off, mask=mask_m[:, None], other=0.0)

            # w: [IC_PADDED, BLOCK_N]
            k_idx = (kh * KW + kw) * IC_PADDED + ic_range  # [IC_PADDED]
            w_off = k_idx[:, None] * stride_wk + offs_n[None, :] * stride_wo
            w_vals = tl.load(w_ptr + w_off, mask=mask_n[None, :], other=0.0)

            acc += tl.dot(x_vals, w_vals, allow_tf32=True)

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    relu6 = tl.minimum(tl.maximum(acc + 3.0, 0.0), 6.0)
    hs = acc * relu6 * (1.0 / 6.0)
    out_val = tl.maximum(hs, 0.0)

    out_off = (b * stride_ob + oh * stride_oh + ow * stride_ow)[:, None] + offs_n[None, :] * stride_oc
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_off, out_val, mask=mask)


def _next_mult(x, m):
    return ((x + m - 1) // m) * m


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-pack weight as [KH*KW*IC_PADDED, OC]
        with torch.no_grad():
            w = self.conv.weight.detach().cuda().contiguous()  # [OC, IC, KH, KW]
            OC, IC, KH, KW = w.shape
            IC_PADDED = max(_next_mult(IC, 16), 16)
            # permute to [KH, KW, IC, OC]
            w_perm = w.permute(2, 3, 1, 0).contiguous()  # [KH, KW, IC, OC]
            w_padded = torch.zeros((KH, KW, IC_PADDED, OC), device=w.device, dtype=w.dtype)
            w_padded[:, :, :IC, :] = w_perm
            w_packed = w_padded.reshape(KH * KW * IC_PADDED, OC).contiguous()
            self.register_buffer('w_packed', w_packed)
            self.register_buffer('bias_buf', self.conv.bias.detach().cuda().contiguous())
            self._IC_PADDED = IC_PADDED
            self._KH = KH
            self._KW = KW
            self._OC = OC
            self._IC = IC

    def forward(self, x):
        x = x.cuda().contiguous()
        B, IC, IH, IW = x.shape
        IC_PADDED = self._IC_PADDED
        KH = self._KH
        KW = self._KW
        OC = self._OC

        # Pad input channels to IC_PADDED
        if IC_PADDED != IC:
            x_padded = torch.zeros((B, IC_PADDED, IH, IW), device=x.device, dtype=x.dtype)
            x_padded[:, :IC, :, :] = x
            x = x_padded.contiguous()

        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((B, OC, OH, OW), device=x.device, dtype=x.dtype)
        N_OUT = B * OH * OW

        grid = lambda meta: (
            triton.cdiv(N_OUT, meta['BLOCK_M']),
            triton.cdiv(OC, meta['BLOCK_N']),
        )

        conv_hswish_relu_kernel[grid](
            x, self.w_packed, self.bias_buf, out,
            B, IC_PADDED, IH, IW,
            OC, OH, OW,
            KH, KW,
            N_OUT,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.w_packed.stride(0), self.w_packed.stride(1),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        )
        return out