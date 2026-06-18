import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=2, num_stages=2),
    ],
    key=['M', 'N', 'K', 'OH', 'OW'],
)
@triton.jit
def conv_relu_hardswish_kernel(
    x_ptr,        # [B, IH, IW, IC]  (NHWC)
    w_ptr,        # [OC, KH*KW*IC]   (row-major, contiguous)
    b_ptr,        # [OC]
    out_ptr,      # [B, OH, OW, OC]  (NHWC)
    B, IH, IW,
    OH, OW,
    IC: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    K: tl.constexpr,   # IC*KH*KW
    M,                  # B*OH*OW
    N: tl.constexpr,   # OC
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    ow = offs_m % OW
    tmp = offs_m // OW
    oh = tmp % OH
    b = tmp // OH

    # K = IC*KH*KW
    offs_k = tl.arange(0, K)  # K is constexpr (=72)
    # decode k -> (kh, kw, ic) where layout is kh-major: k = (kh*KW + kw)*IC + ic
    ic = offs_k % IC
    tmp_k = offs_k // IC
    kw = tmp_k % KW
    kh = tmp_k // KW

    # x is NHWC: stride_b = IH*IW*IC, stride_h = IW*IC, stride_w = IC, stride_c = 1
    ih = oh[:, None] + kh[None, :]
    iw = ow[:, None] + kw[None, :]
    x_offs = (b[:, None] * (IH * IW * IC) +
              ih * (IW * IC) +
              iw * IC +
              ic[None, :])
    x_mask = mask_m[:, None]
    a = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)  # [BLOCK_M, K]

    # w is [OC, K] row-major
    w_offs = offs_n[None, :] * K + offs_k[:, None]
    w_mask = mask_n[None, :]
    bw = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)  # [K, BLOCK_N]

    acc = tl.dot(a, bw, out_dtype=tl.float32)

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # relu
    acc = tl.maximum(acc, 0.0)
    # hardswish: x * clamp((x+3)/6, 0, 1)
    hs = (acc + 3.0) * (1.0 / 6.0)
    hs = tl.minimum(tl.maximum(hs, 0.0), 1.0)
    acc = acc * hs

    # store NHWC out: [B, OH, OW, OC]
    out_offs = (b[:, None] * (OH * OW * N) +
                oh[:, None] * (OW * N) +
                ow[:, None] * N +
                offs_n[None, :])
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-permute weight to [OC, KH, KW, IC] -> flat [OC, KH*KW*IC]
        with torch.no_grad():
            w = self.conv.weight.detach()  # [OC, IC, KH, KW]
            w_perm = w.permute(0, 2, 3, 1).contiguous()  # [OC, KH, KW, IC]
            OC = w_perm.shape[0]
            self.register_buffer('w_packed', w_perm.view(OC, -1).contiguous().cuda())
            self.register_buffer('b_packed', self.conv.bias.detach().contiguous().cuda())

    def forward(self, x):
        x = x.cuda()
        B, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1

        # NHWC layout
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [B, IH, IW, IC]

        out_nhwc = torch.empty((B, OH, OW, OC), device=x.device, dtype=x.dtype)

        M = B * OH * OW
        K = IC * KH * KW
        N = OC

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

        conv_relu_hardswish_kernel[grid](
            x_nhwc, self.w_packed, self.b_packed, out_nhwc,
            B, IH, IW,
            OH, OW,
            IC, KH, KW,
            K, M, N,
        )

        # back to NCHW
        return out_nhwc.permute(0, 3, 1, 2).contiguous()