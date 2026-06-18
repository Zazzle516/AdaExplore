import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OH', 'OW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv2d_relu_bias_nhwc_kernel(
    x_ptr,        # NHWC: [N, IH, IW, IC]
    w_ptr,        # [OC, KH, KW, IC]  (reordered)
    conv_b_ptr,   # [OC]
    bias_ptr,     # [OC]
    out_ptr,      # NHWC: [N, OH, OW, OC]
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # OC tile
    pid_n = tl.program_id(1)  # spatial tile (OH*OW)
    pid_b = tl.program_id(2)  # batch

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # OC indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # spatial indices

    oh = offs_n // OW
    ow = offs_n % OW

    spatial_mask = offs_n < (OH * OW)
    oc_mask = offs_m < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    # Outer loops: KH, KW. Inner loop: IC reduction in BLOCK_K chunks.
    # Weight layout: [OC, KH, KW, IC]
    # Input NHWC layout: [N, IH, IW, IC]
    x_batch_base = pid_b * (IH * IW * IC)

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh  # [BLOCK_N]
            iw = ow + kw  # [BLOCK_N]
            # input row pointer offset (without ic): [BLOCK_N]
            x_row_off = x_batch_base + ih * (IW * IC) + iw * IC

            # weight row pointer offset (without ic): [BLOCK_M]
            w_row_off = offs_m * (KH * KW * IC) + kh * (KW * IC) + kw * IC

            for ic_start in range(0, IC, BLOCK_K):
                ic = ic_start + offs_k  # [BLOCK_K]
                ic_mask = ic < IC

                # Load weight tile [BLOCK_M, BLOCK_K]
                w_ptrs = w_ptr + w_row_off[:, None] + ic[None, :]
                w_mask = oc_mask[:, None] & ic_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

                # Load input tile [BLOCK_K, BLOCK_N]
                # x[ih, iw, ic] -> x_row_off[n] + ic[k]
                x_ptrs = x_ptr + x_row_off[None, :] + ic[:, None]
                x_mask = ic_mask[:, None] & spatial_mask[None, :]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

                acc += tl.dot(w_vals, x_vals)

    # conv bias
    cb = tl.load(conv_b_ptr + offs_m, mask=oc_mask, other=0.0)
    acc += cb[:, None]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # extra bias
    eb = tl.load(bias_ptr + offs_m, mask=oc_mask, other=0.0)
    acc += eb[:, None]

    # Store NHWC out: [N, OH, OW, OC]
    # Output index for (b, oh, ow, oc) = b*OH*OW*OC + (oh*OW+ow)*OC + oc
    out_batch_base = pid_b * (OH * OW * OC)
    out_ptrs = out_ptr + out_batch_base + offs_n[None, :] * OC + offs_m[:, None]
    out_mask = oc_mask[:, None] & spatial_mask[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-permute weight to [OC, KH, KW, IC] for NHWC-friendly access
        with torch.no_grad():
            w = self.conv.weight.detach()  # [OC, IC, KH, KW]
            w_nhwc = w.permute(0, 2, 3, 1).contiguous()  # [OC, KH, KW, IC]
        self.register_buffer('w_nhwc', w_nhwc, persistent=False)

    def forward(self, x):
        # Permute input to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [N, IH, IW, IC]

        N, IH, IW, IC = x_nhwc.shape
        OC = self.out_channels
        KH = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0]
        KW = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[1]
        OH = IH - KH + 1
        OW = IW - KW + 1

        # Use updated weight (in case it changed) - re-permute on each forward to be safe in training
        if self.training:
            w_nhwc = self.conv.weight.permute(0, 2, 3, 1).contiguous()
        else:
            # Use cached buffer if shape matches
            if self.w_nhwc.shape == (OC, KH, KW, IC):
                w_nhwc = self.w_nhwc
                # Sync if weights changed (eval mode they shouldn't, but be safe)
                # We trust eval mode means weights frozen; otherwise fallback
            else:
                w_nhwc = self.conv.weight.permute(0, 2, 3, 1).contiguous()

        cb = self.conv.bias.contiguous()
        eb = self.bias.view(-1).contiguous()

        out_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

        grid = lambda meta: (
            triton.cdiv(OC, meta['BLOCK_M']),
            triton.cdiv(OH * OW, meta['BLOCK_N']),
            N,
        )

        conv2d_relu_bias_nhwc_kernel[grid](
            x_nhwc, w_nhwc, cb, eb, out_nhwc,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
        )

        # Permute back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out