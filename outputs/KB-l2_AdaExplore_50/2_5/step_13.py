import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['OH', 'OW', 'IC', 'OC'],
)
@triton.jit
def conv_transpose2d_kernel(
    x_ptr, w_ptr, b_ptr, eb_ptr, out_ptr,
    N, IC, IH, IW, OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    # strides for x: (IC*IH*IW, IH*IW, IW, 1)
    # strides for w: (OC*KH*KW, KH*KW, KW, 1) -- w shape [IC, OC, KH, KW]
    # strides for out: (OC*OH*OW, OH*OW, OW, 1)
    BLOCK_M: tl.constexpr,  # spatial tile (OH*OW)
    BLOCK_N: tl.constexpr,  # OC tile
    BLOCK_K: tl.constexpr,  # IC tile
):
    pid_n_batch = tl.program_id(0)  # batch index
    pid_m = tl.program_id(1)  # spatial tile id
    pid_oc = tl.program_id(2)  # oc tile id

    n = pid_n_batch

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # spatial positions
    offs_oc = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)  # output channels

    oh = offs_m // OW
    ow = offs_m % OW
    m_mask = offs_m < (OH * OW)
    oc_mask = offs_oc < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over kh, kw, and ic blocks
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            # ih*SH = oh + PH - kh  => ih = (oh + PH - kh)/SH if divisible
            num_h = oh + PH - kh
            num_w = ow + PW - kw
            ih = num_h // SH
            iw = num_w // SW
            valid_h = (num_h % SH == 0) & (ih >= 0) & (ih < IH)
            valid_w = (num_w % SW == 0) & (iw >= 0) & (iw < IW)
            valid = valid_h & valid_w & m_mask  # [BLOCK_M]

            # gather inputs over IC: for each m, need x[n, :, ih, iw]
            # We'll loop over ic in chunks of BLOCK_K
            for ic_start in range(0, IC, BLOCK_K):
                offs_ic = ic_start + tl.arange(0, BLOCK_K)
                ic_mask = offs_ic < IC

                # x_ptrs: [BLOCK_M, BLOCK_K] - x[n, ic, ih, iw]
                x_ptrs = x_ptr + n * (IC * IH * IW) + offs_ic[None, :] * (IH * IW) + ih[:, None] * IW + iw[:, None]
                x_mask = valid[:, None] & ic_mask[None, :]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

                # w_ptrs: [BLOCK_K, BLOCK_N] - w[ic, oc, kh, kw]
                w_ptrs = w_ptr + offs_ic[:, None] * (OC * KH * KW) + offs_oc[None, :] * (KH * KW) + kh * KW + kw
                w_mask = ic_mask[:, None] & oc_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

                acc += tl.dot(x_vals, w_vals)

    # Add conv bias (per OC)
    b_vals = tl.load(b_ptr + offs_oc, mask=oc_mask, other=0.0)  # [BLOCK_N]
    acc += b_vals[None, :]

    # Subtract extra bias (per OC)
    eb_vals = tl.load(eb_ptr + offs_oc, mask=oc_mask, other=0.0)  # [BLOCK_N]
    acc -= eb_vals[None, :]

    # tanh
    # use closed form via exp
    # tanh(x) = (e^{2x} - 1) / (e^{2x} + 1)
    e2x = tl.exp(2.0 * acc)
    acc = (e2x - 1.0) / (e2x + 1.0)

    # Store: out[n, oc, oh, ow]
    out_ptrs = out_ptr + n * (OC * OH * OW) + offs_oc[None, :] * (OH * OW) + offs_m[:, None]
    store_mask = m_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptrs, acc, mask=store_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        # Match the reference init exactly so randomness lines up if seeded
        conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   output_padding=output_padding)
        self.weight = nn.Parameter(conv.weight.data.clone())  # [IC, OC, KH, KW]
        self.conv_bias = nn.Parameter(conv.bias.data.clone())  # [OC]
        self.bias = nn.Parameter(torch.randn(bias_shape))  # [OC, 1, 1]

    def forward(self, x):
        x = x.contiguous()
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        SH = SW = self.stride
        PH = PW = self.padding
        OPH = OPW = self.output_padding

        OH = (IH - 1) * SH - 2 * PH + KH + OPH
        OW = (IW - 1) * SW - 2 * PW + KW + OPW

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        w = self.weight.contiguous()  # [IC, OC, KH, KW]
        cb = self.conv_bias.contiguous()
        eb = self.bias.view(-1).contiguous()

        def grid(meta):
            return (
                N,
                triton.cdiv(OH * OW, meta['BLOCK_M']),
                triton.cdiv(OC, meta['BLOCK_N']),
            )

        conv_transpose2d_kernel[grid](
            x, w, cb, eb, out,
            N, IC, IH, IW, OC, OH, OW,
            KH, KW, SH, SW, PH, PW,
        )
        return out