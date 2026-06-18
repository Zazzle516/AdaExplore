import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_im2col_gemm_kernel(
    x_ptr,   # NHWC input, shape [B, IH, IW, IC]
    w_ptr,   # weight reshaped to [OC, IC*KH*KW] then transposed -> [K, OC] (K = IC*KH*KW)
    b_ptr,
    out_ptr, # NHWC output, shape [B, OH, OW, OC]
    B, IH, IW, IC,
    OH, OW, OC,
    KH: tl.constexpr, KW: tl.constexpr, IC_C: tl.constexpr,
    M, N, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows of output (B*OH*OW)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # OC

    mask_m = offs_m < M
    mask_n = offs_n < N

    # decode m -> (b, oh, ow)
    ow = offs_m % OW
    tmp = offs_m // OW
    oh = tmp % OH
    b  = tmp // OH

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate K dimension as kh, kw, ic. We tile K by IC_C (full IC inner).
    # K = KH * KW * IC. Inner block over IC (contiguous in NHWC).
    offs_ic = tl.arange(0, IC_C)  # [IC_C]

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh  # [BLOCK_M]
            iw = ow + kw  # [BLOCK_M]
            # x offset: ((b*IH + ih)*IW + iw)*IC + ic
            x_base = ((b * IH + ih) * IW + iw) * IC  # [BLOCK_M]
            x_off = x_base[:, None] + offs_ic[None, :]  # [BLOCK_M, IC_C]
            x_mask = mask_m[:, None] & (offs_ic[None, :] < IC)
            x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [BLOCK_M, IC_C]

            # weight is stored as [K, OC] where K iterates (kh, kw, ic)
            # k_idx = (kh*KW + kw)*IC + ic
            k_base = (kh * KW + kw) * IC
            w_off = (k_base + offs_ic[:, None]) * N + offs_n[None, :]  # [IC_C, BLOCK_N]
            w_mask = (offs_ic[:, None] < IC) & mask_n[None, :]
            w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # [IC_C, BLOCK_N]

            acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # relu(hardswish(x)): hardswish(x)=x*relu6(x+3)/6, relu of that
    relu6 = tl.minimum(tl.maximum(acc + 3.0, 0.0), 6.0)
    hs = acc * relu6 * (1.0 / 6.0)
    out_val = tl.maximum(hs, 0.0)

    # store NHWC: out[b, oh, ow, oc]
    out_off = (((b * OH + oh) * OW + ow) * OC)[:, None] + offs_n[None, :]
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_off, out_val, mask=mask)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]

        # Pre-reshape weight: [OC, IC, KH, KW] -> permute to [KH, KW, IC, OC] -> [K, OC]
        with torch.no_grad():
            w = self.conv.weight.detach()  # [OC, IC, KH, KW]
            OC, IC, KH, KW = w.shape
            w_reshaped = w.permute(2, 3, 1, 0).contiguous().view(KH * KW * IC, OC)
        self.register_buffer('w_packed', w_reshaped.cuda())
        self.register_buffer('b_packed', self.conv.bias.detach().cuda().contiguous())
        self._KH = KH
        self._KW = KW
        self._IC = IC
        self._OC = OC
        self._IC_C = _next_pow2(IC)

    def forward(self, x):
        x = x.cuda()
        # to NHWC
        B, IC, IH, IW = x.shape
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        KH = self._KH
        KW = self._KW
        OC = self._OC
        OH = IH - KH + 1
        OW = IW - KW + 1

        out_nhwc = torch.empty((B, OH, OW, OC), device=x.device, dtype=x.dtype)

        M = B * OH * OW
        N = OC
        K = KH * KW * IC

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

        conv_im2col_gemm_kernel[grid](
            x_nhwc, self.w_packed, self.b_packed, out_nhwc,
            B, IH, IW, IC,
            OH, OW, OC,
            KH, KW, self._IC_C,
            M, N, K,
        )

        # back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out