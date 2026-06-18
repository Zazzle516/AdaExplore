import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=2, num_stages=3),
    ],
    key=['M', 'N', 'K', 'OH', 'OW'],
)
@triton.jit
def conv_implicit_gemm_kernel(
    x_ptr,         # [B, IH, IW, IC] NHWC
    w_ptr,         # [OC, KH, KW, IC]  -> viewed as [N=OC, K=KH*KW*IC]
    b_ptr,         # [OC]
    y_ptr,         # [B, OH, OW, OC] NHWC
    B, IH, IW, IC,
    OH, OW, OC,
    KH: tl.constexpr, KW: tl.constexpr,
    M, N, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # output row positions (B*OH*OW)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # OC

    mask_m = offs_m < M
    mask_n = offs_n < N

    # decompose m -> (b, oh, ow)
    ow = offs_m % OW
    tmp = offs_m // OW
    oh = tmp % OH
    b = tmp // OH

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, K)
    # k -> (kh, kw, ic)
    ic_k = offs_k % IC
    tmp_k = offs_k // IC
    kw_k = tmp_k % KW
    kh_k = tmp_k // KW

    # x offsets: for each (m, k): ((b*IH + (oh+kh))*IW + (ow+kw))*IC + ic
    # Build [BLOCK_M, K] index
    ih = oh[:, None] + kh_k[None, :]   # [BLOCK_M, K]
    iw = ow[:, None] + kw_k[None, :]   # [BLOCK_M, K]
    x_offs = ((b[:, None] * IH + ih) * IW + iw) * IC + ic_k[None, :]

    x_mask = mask_m[:, None]
    x_vals = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)  # [BLOCK_M, K]

    # w offsets: [OC, KH*KW*IC] = [N, K], rows = offs_n
    w_offs = offs_n[:, None] * K + offs_k[None, :]  # [BLOCK_N, K]
    w_mask = mask_n[:, None]
    w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)  # [BLOCK_N, K]

    # acc += x_vals @ w_vals.T   -> [BLOCK_M, BLOCK_N]
    acc = tl.dot(x_vals, tl.trans(w_vals), acc=acc, out_dtype=tl.float32)

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # ReLU + HardSwish: x * clamp((x+3)/6, 0, 1)
    acc = tl.maximum(acc, 0.0)
    hs = (acc + 3.0) * (1.0 / 6.0)
    hs = tl.minimum(tl.maximum(hs, 0.0), 1.0)
    out = acc * hs

    # store [BLOCK_M, BLOCK_N] into y NHWC: y[b, oh, ow, oc]
    y_offs = ((b[:, None] * OH + oh[:, None]) * OW + ow[:, None]) * OC + offs_n[None, :]
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptr + y_offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Preprocess weight: [OC, IC, KH, KW] -> [OC, KH, KW, IC] -> flat [OC, KH*KW*IC]
        with torch.no_grad():
            w = self.conv.weight.detach().cuda().contiguous()
            OC, IC, KH, KW = w.shape
            w_perm = w.permute(0, 2, 3, 1).contiguous().view(OC, KH * KW * IC)
            self.register_buffer('w_packed', w_perm)
            self.register_buffer('b_packed', self.conv.bias.detach().cuda().contiguous())
            self._KH = KH
            self._KW = KW
            self._OC = OC
            self._IC = IC

    def forward(self, x):
        x = x.cuda()
        # NCHW -> NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        B, IH, IW, IC = x_nhwc.shape
        KH = self._KH
        KW = self._KW
        OC = self._OC
        OH = IH - KH + 1
        OW = IW - KW + 1
        K = KH * KW * IC

        y_nhwc = torch.empty((B, OH, OW, OC), device=x.device, dtype=x.dtype)

        M = B * OH * OW
        N = OC

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

        conv_implicit_gemm_kernel[grid](
            x_nhwc, self.w_packed, self.b_packed, y_nhwc,
            B, IH, IW, IC,
            OH, OW, OC,
            KH, KW,
            M, N, K,
        )

        # NHWC -> NCHW
        return y_nhwc.permute(0, 3, 1, 2).contiguous()