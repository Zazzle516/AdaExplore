import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OH', 'OW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv2d_relu_bias_nhwc_kernel(
    x_ptr,        # NHWC: [N, IH, IW, IC]
    w_ptr,        # [OC, KH, KW, IC]
    conv_b_ptr,   # [OC]
    bias_ptr,     # [OC]
    out_ptr,      # NHWC: [N, OH, OW, OC]
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # OC tile
    pid_n = tl.program_id(1)  # spatial tile (OW-contiguous)
    pid_bh = tl.program_id(2)  # batch * OH

    pid_b = pid_bh // OH
    oh = pid_bh % OH

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # OC indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # OW indices

    oc_mask = offs_m < OC
    ow_mask = offs_n < OW

    offs_ic = tl.arange(0, BLOCK_IC)
    ic_mask = offs_ic < IC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x_batch_base = pid_b * (IH * IW * IC)

    for kh in tl.static_range(0, KH):
        ih = oh + kh
        for kw in tl.static_range(0, KW):
            iw = offs_n + kw  # [BLOCK_N]
            # input pointer offset for spatial: x_batch_base + ih*IW*IC + iw*IC
            x_row_off = x_batch_base + ih * (IW * IC) + iw * IC  # [BLOCK_N]

            # weight offset: oc * KH*KW*IC + kh*KW*IC + kw*IC + ic
            w_row_off = offs_m * (KH * KW * IC) + kh * (KW * IC) + kw * IC  # [BLOCK_M]

            # Load weight [BLOCK_M, BLOCK_IC]
            w_ptrs = w_ptr + w_row_off[:, None] + offs_ic[None, :]
            w_mask = oc_mask[:, None] & ic_mask[None, :]
            w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

            # Load input [BLOCK_IC, BLOCK_N]
            x_ptrs = x_ptr + x_row_off[None, :] + offs_ic[:, None]
            x_mask = ic_mask[:, None] & ow_mask[None, :]
            x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

            acc += tl.dot(w_vals, x_vals)

    cb = tl.load(conv_b_ptr + offs_m, mask=oc_mask, other=0.0)
    acc += cb[:, None]

    acc = tl.maximum(acc, 0.0)

    eb = tl.load(bias_ptr + offs_m, mask=oc_mask, other=0.0)
    acc += eb[:, None]

    # Store: [N, OH, OW, OC]
    out_batch_base = pid_b * (OH * OW * OC) + oh * (OW * OC)
    out_ptrs = out_ptr + out_batch_base + offs_n[None, :] * OC + offs_m[:, None]
    out_mask = oc_mask[:, None] & ow_mask[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        with torch.no_grad():
            w = self.conv.weight.detach()
            w_nhwc = w.permute(0, 2, 3, 1).contiguous()
        self.register_buffer('w_nhwc_cache', w_nhwc, persistent=False)
        self._cached_weight_version = self.conv.weight._version

    def _get_w_nhwc(self):
        if self.training or self.conv.weight._version != self._cached_weight_version:
            w_nhwc = self.conv.weight.permute(0, 2, 3, 1).contiguous()
            if not self.training:
                self.w_nhwc_cache = w_nhwc
                self._cached_weight_version = self.conv.weight._version
            return w_nhwc
        return self.w_nhwc_cache

    def forward(self, x):
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        N, IH, IW, IC = x_nhwc.shape
        OC = self.out_channels
        KH = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0]
        KW = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[1]
        OH = IH - KH + 1
        OW = IW - KW + 1

        w_nhwc = self._get_w_nhwc()
        cb = self.conv.bias.contiguous()
        eb = self.bias.view(-1).contiguous()

        out_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

        # Round IC up to next power-of-2 for BLOCK_IC
        BLOCK_IC = 1
        while BLOCK_IC < IC:
            BLOCK_IC *= 2

        grid = lambda meta: (
            triton.cdiv(OC, meta['BLOCK_M']),
            triton.cdiv(OW, meta['BLOCK_N']),
            N * OH,
        )

        conv2d_relu_bias_nhwc_kernel[grid](
            x_nhwc, w_nhwc, cb, eb, out_nhwc,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            BLOCK_IC,
        )

        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out