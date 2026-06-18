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
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_gemm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, IH, IW, IC,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    M, N, K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # x layout: NHWC, contiguous on IC fastest
    # w layout: [OC, KH*KW*IC] = [N, K]
    # out layout: NHWC, contiguous on OC fastest, shape [B, OH, OW, OC]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # along M = B*OH*OW
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # along N = OC

    mask_m = offs_m < M
    mask_n = offs_n < N

    # decode M -> (b, oh, ow)
    ow = offs_m % OW
    tmp = offs_m // OW
    oh = tmp % OH
    b = tmp // OH

    # K dim layout: k = kh*KW*IC + kw*IC + ic
    offs_k = tl.arange(0, K)
    ic_k = offs_k % IC
    tmp_k = offs_k // IC
    kw_k = tmp_k % KW
    kh_k = tmp_k // KW

    # input pointer per (m, k)
    ih = oh[:, None] + kh_k[None, :]  # [BLOCK_M, K]
    iw = ow[:, None] + kw_k[None, :]
    # x is NHWC contiguous: offset = b*IH*IW*IC + ih*IW*IC + iw*IC + ic
    x_offs = (b[:, None] * (IH * IW * IC)
              + ih * (IW * IC)
              + iw * IC
              + ic_k[None, :])
    x_tile = tl.load(x_ptr + x_offs, mask=mask_m[:, None], other=0.0)  # [BLOCK_M, K]

    # weight: [OC, K], contiguous on K
    w_offs = offs_n[:, None] * K + offs_k[None, :]  # [BLOCK_N, K]
    w_tile = tl.load(w_ptr + w_offs, mask=mask_n[:, None], other=0.0)  # [BLOCK_N, K]

    # We want acc = x_tile @ w_tile.T : [BLOCK_M, BLOCK_N]
    acc = tl.dot(x_tile, tl.trans(w_tile))

    # bias
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # relu
    acc = tl.maximum(acc, 0.0)
    # hardswish
    hs = (acc + 3.0) * (1.0 / 6.0)
    hs = tl.minimum(tl.maximum(hs, 0.0), 1.0)
    acc = acc * hs

    # store NHWC: out_offs = b*OH*OW*OC + oh*OW*OC + ow*OC + oc
    out_offs = (b[:, None] * (OH * OW * OC)
                + oh[:, None] * (OW * OC)
                + ow[:, None] * OC
                + offs_n[None, :])
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self._cached_w = None
        self._cached_b = None

    def _prep_weights(self, device, dtype):
        # weight in pytorch: [OC, IC, KH, KW]
        # rearrange to [OC, KH, KW, IC] then flatten -> [OC, KH*KW*IC]
        w = self.conv.weight.detach().to(device=device, dtype=dtype)
        OC, IC, KH, KW = w.shape
        w_perm = w.permute(0, 2, 3, 1).contiguous().view(OC, KH * KW * IC)
        b = self.conv.bias.detach().to(device=device, dtype=dtype).contiguous()
        return w_perm, b

    def forward(self, x):
        x = x.cuda()
        device = x.device
        dtype = x.dtype

        if (self._cached_w is None or self._cached_w.device != device
                or self._cached_w.dtype != dtype):
            self._cached_w, self._cached_b = self._prep_weights(device, dtype)

        w = self._cached_w
        b = self._cached_b

        B, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1

        # convert input to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        # allocate output in NHWC
        out_nhwc = torch.empty((B, OH, OW, OC), device=device, dtype=dtype)

        M = B * OH * OW
        N = OC
        K = IC * KH * KW

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

        conv_gemm_kernel[grid](
            x_nhwc, w, b, out_nhwc,
            B, IH, IW, IC,
            OC, OH, OW,
            KH, KW,
            M, N, K,
        )

        # convert back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out