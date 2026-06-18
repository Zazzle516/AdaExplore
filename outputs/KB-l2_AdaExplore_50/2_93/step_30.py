import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_fused_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr,
    add_value: tl.constexpr,
    multiply_value: tl.constexpr,
    BLOCK_M: tl.constexpr,  # spatial tile
    BLOCK_N: tl.constexpr,  # OC tile
    BLOCK_K: tl.constexpr,  # IC tile
):
    pid_n = tl.program_id(0)  # batch
    pid_oc = tl.program_id(1)  # OC tile
    pid_sp = tl.program_id(2)  # spatial tile

    # Output spatial offsets within tile
    sp_offs = pid_sp * BLOCK_M + tl.arange(0, BLOCK_M)
    oc_offs = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)

    oh = sp_offs // OW
    ow = sp_offs % OW

    sp_mask = sp_offs < (OH * OW)
    oc_mask = oc_offs < OC

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # For ConvTranspose2d:
    # out[n, oc, oh, ow] = sum_{ic, kh, kw} x[n, ic, ih, iw] * w[ic, oc, kh, kw]
    # where ih = (oh - kh) / stride if (oh - kh) % stride == 0 else skip
    # Same for iw.
    # Loop over kh, kw, ic-tiles.

    # n batch index
    n = pid_n

    for kh in tl.static_range(0, KH):
        # ih_num = oh - kh; valid if ih_num >= 0 and ih_num % STRIDE == 0 and ih_num/STRIDE < IH
        ih_num = oh - kh
        ih = ih_num // STRIDE
        ih_valid = (ih_num >= 0) & ((ih_num % STRIDE) == 0) & (ih < IH) & (ih_num >= 0)
        for kw in tl.static_range(0, KW):
            iw_num = ow - kw
            iw = iw_num // STRIDE
            iw_valid = (iw_num >= 0) & ((iw_num % STRIDE) == 0) & (iw < IW)
            spatial_valid = ih_valid & iw_valid & sp_mask  # [BLOCK_M]

            # Loop over IC in tiles
            for ic_start in range(0, IC, BLOCK_K):
                ic_offs = ic_start + tl.arange(0, BLOCK_K)
                ic_mask = ic_offs < IC

                # Load x[n, ic_offs, ih, iw] -> shape [BLOCK_M, BLOCK_K]
                # input index: n * (IC*IH*IW) + ic * (IH*IW) + ih * IW + iw
                x_offs = (n * IC * IH * IW
                          + ic_offs[None, :] * (IH * IW)
                          + ih[:, None] * IW
                          + iw[:, None])
                x_mask = spatial_valid[:, None] & ic_mask[None, :]
                x_vals = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

                # Load w[ic_offs, oc_offs, kh, kw] -> shape [BLOCK_K, BLOCK_N]
                # weight index: ic * (OC*KH*KW) + oc * (KH*KW) + kh * KW + kw
                w_offs = (ic_offs[:, None] * (OC * KH * KW)
                          + oc_offs[None, :] * (KH * KW)
                          + kh * KW + kw)
                w_mask = ic_mask[:, None] & oc_mask[None, :]
                w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals)

    # Add bias
    b = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + b[None, :] + add_value

    # min(v, 0)
    acc = tl.minimum(acc, 0.0)

    # GELU exact: 0.5 * v * (1 + erf(v / sqrt(2)))
    inv_sqrt2 = 0.70710678118654752440
    acc = 0.5 * acc * (1.0 + tl.math.erf(acc * inv_sqrt2))

    # multiply
    acc = acc * multiply_value

    # Store: out[n, oc, oh, ow]
    out_offs = (n * OC * OH * OW
                + oc_offs[None, :] * (OH * OW)
                + oh[:, None] * OW + ow[:, None])
    out_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = float(add_value)
        self.multiply_value = float(multiply_value)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()
        bias = self.conv_transpose.bias.contiguous()

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        STRIDE = self.stride
        # ConvTranspose2d output size: (IH - 1)*stride - 2*padding + KH (+ output_padding)
        OH = (IH - 1) * STRIDE + KH
        OW = (IW - 1) * STRIDE + KW

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid = (N, triton.cdiv(OC, BLOCK_N), triton.cdiv(OH * OW, BLOCK_M))

        conv_transpose_fused_kernel[grid](
            x, w, bias, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH=KH, KW=KW,
            STRIDE=STRIDE,
            add_value=self.add_value,
            multiply_value=self.multiply_value,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=2,
        )
        return out