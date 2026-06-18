import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _convt_gather_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,  # output spatial tile
    BLOCK_N: tl.constexpr,  # OC tile
    BLOCK_K: tl.constexpr,  # IC tile
):
    pid_m = tl.program_id(0)  # output spatial block
    pid_n = tl.program_id(1)  # OC block
    pid_b = tl.program_id(2)  # batch index

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # output spatial positions
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # OC indices

    oh = offs_m // OW
    ow = offs_m % OW
    m_mask = offs_m < (OH * OW)
    n_mask = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # For each kernel position
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            # input position: ih = (oh + pad - kh) / stride, must be int and >=0 and < IH
            ih_num = oh + PAD - kh
            iw_num = ow + PAD - kw
            ih = ih_num // STRIDE
            iw = iw_num // STRIDE
            valid_h = (ih_num % STRIDE == 0) & (ih >= 0) & (ih < IH)
            valid_w = (iw_num % STRIDE == 0) & (iw >= 0) & (iw < IW)
            valid = valid_h & valid_w & m_mask  # [BLOCK_M]

            # accumulate over IC
            for ic_start in range(0, IC, BLOCK_K):
                offs_k = ic_start + tl.arange(0, BLOCK_K)
                k_mask = offs_k < IC  # [BLOCK_K]

                # Load x[b, ic, ih, iw] -> shape [BLOCK_M, BLOCK_K]
                x_offs = (pid_b * IC * IH * IW
                          + offs_k[None, :] * IH * IW
                          + ih[:, None] * IW
                          + iw[:, None])
                x_mask = valid[:, None] & k_mask[None, :]
                x_vals = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

                # Load w[ic, oc, kh, kw] -> shape [BLOCK_K, BLOCK_N]
                w_offs = (offs_k[:, None] * OC * KH * KW
                          + offs_n[None, :] * KH * KW
                          + kh * KW + kw)
                w_mask = k_mask[:, None] & n_mask[None, :]
                w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals)

    # epilogue: subtract fused bias and tanh
    b = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc - b[None, :]
    e2x = tl.exp(2.0 * acc)
    out = (e2x - 1.0) / (e2x + 1.0)

    out_offs = (pid_b * OC * OH * OW
                + offs_n[None, :] * OH * OW
                + offs_m[:, None])
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_offs, out, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def _fused_bias(self):
        # conv_transpose.bias: [OC], self.bias: [OC,1,1]
        cb = self.conv_transpose.bias
        sb = self.bias.view(-1)
        return (sb - cb).contiguous()

    def forward(self, x):
        x = x.contiguous()
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        S = self.stride
        P = self.padding
        OP = self.output_padding

        OH = (IH - 1) * S - 2 * P + KH + OP
        OW = (IW - 1) * S - 2 * P + KW + OP

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        weight = self.conv_transpose.weight.contiguous()  # [IC, OC, KH, KW]
        fb = self._fused_bias()  # [OC]

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid = (
            (OH * OW + BLOCK_M - 1) // BLOCK_M,
            (OC + BLOCK_N - 1) // BLOCK_N,
            N,
        )

        _convt_gather_kernel[grid](
            x, weight, fb, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH=KH, KW=KW,
            STRIDE=S, PAD=P,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )
        return out