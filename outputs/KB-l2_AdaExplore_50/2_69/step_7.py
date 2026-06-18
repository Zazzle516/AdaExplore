import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
    ],
    key=['N_OUT', 'OC', 'K'],
)
@triton.jit
def conv_hswish_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, IH, IW,
    OC, OH, OW,
    IC: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    K: tl.constexpr,  # IC*KH*KW
    N_OUT,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # output spatial index (over B*OH*OW)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # OC

    mask_m = offs_m < N_OUT
    mask_n = offs_n < OC

    # decode m -> (b, oh, ow)
    ow = offs_m % OW
    tmp = offs_m // OW
    oh = tmp % OH
    b = tmp // OH

    # x is NHWC: stride_b = IH*IW*IC, stride_h = IW*IC, stride_w = IC, stride_c = 1
    # base offset of (b, oh, ow) in the input -> (b, ih=oh, iw=ow, c=0)
    x_base = b * (IH * IW * IC) + oh * (IW * IC) + ow * IC  # [BLOCK_M]

    # weight layout: (OC, KH, KW, IC) contiguous -> stride_oc = K, stride_kh = KW*IC, stride_kw=IC, stride_ic=1
    # k index: k = (kh*KW + kw)*IC + ic
    offs_k = tl.arange(0, K)
    # decompose k
    ic_k = offs_k % IC
    khw_k = offs_k // IC
    kw_k = khw_k % KW
    kh_k = khw_k // KW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Build a-tile [BLOCK_M, K]: for each m, for each k, gather x[b, oh+kh, ow+kw, ic]
    # offset relative to x_base: kh*(IW*IC) + kw*IC + ic
    a_off = x_base[:, None] + (kh_k[None, :] * (IW * IC) + kw_k[None, :] * IC + ic_k[None, :])
    a_tile = tl.load(x_ptr + a_off, mask=mask_m[:, None], other=0.0)

    # Build b-tile [K, BLOCK_N]: weight[oc, k]
    w_off = offs_n[None, :] * K + offs_k[:, None]
    b_tile = tl.load(w_ptr + w_off, mask=mask_n[None, :], other=0.0)

    acc = tl.dot(a_tile, b_tile, acc=acc)

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    relu6 = tl.minimum(tl.maximum(acc + 3.0, 0.0), 6.0)
    hs = acc * relu6 * (1.0 / 6.0)
    out_val = tl.maximum(hs, 0.0)

    # out is NCHW contiguous
    # out[b, oc, oh, ow] -> b*OC*OH*OW + oc*OH*OW + oh*OW + ow
    out_off = (b * (OC * OH * OW) + oh * OW + ow)[:, None] + offs_n[None, :] * (OH * OW)
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_off, out_val, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-permute weight to (OC, KH, KW, IC) for contiguous K-dim loading
        with torch.no_grad():
            w = self.conv.weight.detach().cuda().contiguous()
            w_perm = w.permute(0, 2, 3, 1).contiguous()  # (OC, KH, KW, IC)
            self.register_buffer('w_packed', w_perm)
            self.register_buffer('b_packed', self.conv.bias.detach().cuda().contiguous())

    def forward(self, x):
        x = x.cuda()
        # NCHW -> NHWC
        if not x.is_contiguous():
            x = x.contiguous()
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        B, IH, IW, IC = x_nhwc.shape
        OC, KH, KW, _ = self.w_packed.shape
        OH = IH - KH + 1
        OW = IW - KW + 1
        K = IC * KH * KW

        out = torch.empty((B, OC, OH, OW), device=x.device, dtype=x.dtype)

        N_OUT = B * OH * OW

        grid = lambda meta: (
            triton.cdiv(N_OUT, meta['BLOCK_M']),
            triton.cdiv(OC, meta['BLOCK_N']),
        )

        conv_hswish_relu_kernel[grid](
            x_nhwc, self.w_packed, self.b_packed, out,
            B, IH, IW,
            OC, OH, OW,
            IC, KH, KW, K,
            N_OUT,
        )
        return out