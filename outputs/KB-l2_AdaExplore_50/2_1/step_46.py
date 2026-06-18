import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OUT_HW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv_relu_bias_kernel(
    x_ptr, w_ptr, b_ptr, bias_add_ptr, out_ptr,
    N, IC, H, W,
    OC, OH, OW,
    OUT_HW, N_OUT_HW,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_M: tl.constexpr,   # spatial (output positions) tile
    BLOCK_N: tl.constexpr,   # OC tile
    BLOCK_K: tl.constexpr,   # IC tile
):
    # NHWC layout
    # x: (N, H, W, IC)   stride = (H*W*IC, W*IC, IC, 1)
    # w: (OC, KH, KW, IC) stride = (KH*KW*IC, KW*IC, IC, 1)
    # out: (N, OH, OW, OC)

    pid_m = tl.program_id(0)   # spatial tile id
    pid_n = tl.program_id(1)   # oc tile id

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # spatial positions
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # OC

    n_idx = offs_m // OUT_HW
    hw_idx = offs_m % OUT_HW
    oh = hw_idx // OW
    ow = hw_idx % OW

    mask_m = offs_m < N_OUT_HW
    mask_n = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    # base x offset per output point (without ic, kh, kw)
    x_base = n_idx * (H * W * IC) + oh * (W * IC) + ow * IC   # [BLOCK_M]
    # base w offset per oc (without ic, kh, kw)
    w_base = offs_n * (KH * KW * IC)                          # [BLOCK_N]

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            # input position offset add for this (kh, kw)
            x_pos = x_base + kh * (W * IC) + kw * IC          # [BLOCK_M]
            w_pos = w_base + kh * (KW * IC) + kw * IC         # [BLOCK_N]
            for ic_start in range(0, IC, BLOCK_K):
                ic = ic_start + offs_k
                mask_k = ic < IC

                # x tile: [BLOCK_M, BLOCK_K]
                x_off = x_pos[:, None] + ic[None, :]
                x_mask = mask_m[:, None] & mask_k[None, :]
                x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                # w tile: [BLOCK_K, BLOCK_N]
                w_off = w_pos[None, :] + ic[:, None]
                w_mask = mask_k[:, None] & mask_n[None, :]
                w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

                acc += tl.dot(x_tile, w_tile)

    # bias from conv [OC]
    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # add learned bias (shape [OC])
    bias_add = tl.load(bias_add_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias_add[None, :]

    # store NHWC
    out_off = (n_idx[:, None] * (OUT_HW * OC)
               + oh[:, None] * (OW * OC)
               + ow[:, None] * OC
               + offs_n[None, :])
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self._cached_w_nhwc = None
        self._cached_w_version = None

    def _get_w_nhwc(self):
        w = self.conv.weight
        if (self._cached_w_nhwc is None
                or self._cached_w_version != w._version
                or self._cached_w_nhwc.device != w.device):
            self._cached_w_nhwc = w.detach().permute(0, 2, 3, 1).contiguous()
            self._cached_w_version = w._version
        return self._cached_w_nhwc

    def forward(self, x):
        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = self.conv.kernel_size[0]
        KW = self.conv.kernel_size[1]
        OH = H - KH + 1
        OW = W - KW + 1

        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        if self.training:
            w_nhwc = self.conv.weight.permute(0, 2, 3, 1).contiguous()
        else:
            w_nhwc = self._get_w_nhwc()

        cb = self.conv.bias.contiguous()
        bias_add = self.bias.contiguous().view(-1)

        out_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

        OUT_HW = OH * OW
        N_OUT_HW = N * OUT_HW

        # choose BLOCK_K: use full IC if it's a power of 2 and reasonable
        BLOCK_K = 64 if IC >= 64 else 32

        grid = lambda meta: (
            triton.cdiv(N_OUT_HW, meta['BLOCK_M']),
            triton.cdiv(OC, meta['BLOCK_N']),
        )

        conv_relu_bias_kernel[grid](
            x_nhwc, w_nhwc, cb, bias_add, out_nhwc,
            N, IC, H, W,
            OC, OH, OW,
            OUT_HW, N_OUT_HW,
            KH, KW,
            BLOCK_K=BLOCK_K,
        )

        return out_nhwc.permute(0, 3, 1, 2).contiguous()