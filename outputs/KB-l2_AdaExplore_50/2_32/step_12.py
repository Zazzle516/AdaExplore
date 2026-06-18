import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OUT_HW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv_scale_min_kernel_fused(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    OH, OW,
    SCALE,
    OUT_HW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)  # batch index
    pid_m = tl.program_id(1)  # tile along output spatial (OH*OW)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < OUT_HW

    oh = offs_m // OW
    ow = offs_m % OW

    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)  # OC channels (full)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # outer loop over (kh, kw), inner over IC in BLOCK_K chunks
    for kh in range(0, KH):
        for kw in range(0, KW):
            ih = oh + kh
            iw = ow + kw
            # x base for this (kh,kw): pid_n*H*W*IC + ih*W*IC + iw*IC
            x_base = pid_n * H * W * IC + ih * (W * IC) + iw * IC  # [BLOCK_M]
            # w base for this (kh,kw): kh*KW*IC*OC + kw*IC*OC
            w_base = kh * (KW * IC * OC) + kw * (IC * OC)

            for ic_start in range(0, IC, BLOCK_K):
                ic = ic_start + offs_k
                k_mask = ic < IC

                x_offset = x_base[:, None] + ic[None, :]
                x_load_mask = m_mask[:, None] & k_mask[None, :]
                x_vals = tl.load(x_ptr + x_offset, mask=x_load_mask, other=0.0)

                w_offset = w_base + ic[:, None] * OC + offs_n[None, :]
                w_vals = tl.load(w_ptr + w_offset, mask=k_mask[:, None], other=0.0)

                acc += tl.dot(x_vals, w_vals)

    bias = tl.load(b_ptr + offs_n)
    acc = acc + bias[None, :]
    acc = acc * SCALE

    min_val = tl.min(acc, axis=1)

    out_offset = pid_n * OUT_HW + offs_m
    tl.store(out_ptr + out_offset, min_val, mask=m_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = float(scale_factor)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        with torch.no_grad():
            w = self.conv.weight.detach()  # (OC, IC, KH, KW)
            w_nhwc = w.permute(2, 3, 1, 0).contiguous()  # (KH, KW, IC, OC)
        self.register_buffer('w_nhwc', w_nhwc, persistent=False)

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, H, W = x.shape
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        w_nhwc = self.w_nhwc
        if w_nhwc.device != x.device:
            w_nhwc = w_nhwc.to(x.device)
            self.w_nhwc = w_nhwc

        b = self.conv.bias.contiguous().to(x.device)

        OC = self.out_channels
        KH = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0]
        KW = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[1]
        OH = H - KH + 1
        OW = W - KW + 1
        OUT_HW = OH * OW

        out = torch.empty((N, 1, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_N = OC  # full OC

        grid = lambda meta: (N, (OUT_HW + meta['BLOCK_M'] - 1) // meta['BLOCK_M'])

        conv_scale_min_kernel_fused[grid](
            x_nhwc, w_nhwc, b, out,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW,
            self.scale_factor,
            OUT_HW,
            BLOCK_N=BLOCK_N,
        )
        return out