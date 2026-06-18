import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16}, num_warps=4, num_stages=3),
    ],
    key=['OC', 'OUT_HW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv_scale_min_kernel_chlast(
    x_ptr,         # [N, H, W, IC]  channels_last
    w_ptr,         # [OC, KH, KW, IC]  channels_last
    b_ptr,         # [OC]
    out_ptr,       # [N, OH, OW]
    N, IC, H, W,
    OC, KH, KW,
    OH, OW,
    scale,
    OUT_HW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    KH_C: tl.constexpr,
    KW_C: tl.constexpr,
):
    pid_n = tl.program_id(0)  # batch
    pid_m = tl.program_id(1)  # output spatial tile

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)   # OC
    offs_k = tl.arange(0, BLOCK_K)   # IC

    mask_m = offs_m < OUT_HW

    oh = offs_m // OW
    ow = offs_m % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # x is [N, H, W, IC]; stride per pixel = IC; per row = W*IC
    x_batch_base = pid_n * H * W * IC

    # For each (kh, kw), gather IC-wide vectors and do tl.dot
    for kh in tl.static_range(0, KH_C):
        for kw in tl.static_range(0, KW_C):
            # input row/col
            ih = oh + kh
            iw = ow + kw
            # offset into x: [N, H, W, IC]
            x_pix_off = x_batch_base + ih * (W * IC) + iw * IC  # [BLOCK_M]
            x_off = x_pix_off[:, None] + offs_k[None, :]         # [BLOCK_M, BLOCK_K]
            x_vals = tl.load(x_ptr + x_off, mask=mask_m[:, None], other=0.0)

            # weight offset: [OC, KH, KW, IC]
            w_base = (kh * KW + kw) * IC
            w_off = offs_n[:, None] * (KH * KW * IC) + w_base + offs_k[None, :]  # [BLOCK_N, BLOCK_K]
            w_vals = tl.load(w_ptr + w_off)

            # acc[BM,BN] += x[BM,BK] @ w[BN,BK].T
            acc += tl.dot(x_vals, tl.trans(w_vals))

    bias = tl.load(b_ptr + offs_n)
    acc = acc + bias[None, :]
    acc = acc * scale

    min_val = tl.min(acc, axis=1)  # [BLOCK_M]

    out_offset = pid_n * OUT_HW + offs_m
    tl.store(out_ptr + out_offset, min_val, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # Pre-transpose weight to channels_last [OC, KH, KW, IC]
        self._cached_w = None
        self._cached_b = None

    def _get_weight(self):
        if self._cached_w is None or self._cached_w.device != self.conv.weight.device:
            w = self.conv.weight.detach()  # [OC, IC, KH, KW]
            w_cl = w.permute(0, 2, 3, 1).contiguous().cuda()
            b = self.conv.bias.detach().contiguous().cuda()
            self._cached_w = w_cl
            self._cached_b = b
        return self._cached_w, self._cached_b

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1
        OUT_HW = OH * OW

        # Transpose input to channels_last [N, H, W, IC]
        x_cl = x.permute(0, 2, 3, 1).contiguous()

        w_cl, b = self._get_weight()

        out = torch.empty((N, 1, OH, OW), device=x.device, dtype=torch.float32)

        BLOCK_N = OC   # 128
        BLOCK_K = IC   # 64

        grid = lambda meta: (N, triton.cdiv(OUT_HW, meta['BLOCK_M']))

        conv_scale_min_kernel_chlast[grid](
            x_cl, w_cl, b, out,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW,
            float(self.scale_factor),
            OUT_HW,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            KH_C=KH,
            KW_C=KW,
        )

        return out