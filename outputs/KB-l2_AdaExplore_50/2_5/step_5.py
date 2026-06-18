import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Gather-form ConvTranspose2d:
# Output shape: (N, OC, H_out, W_out)
# For each output pixel (oh, ow), iterate over (ic, kh, kw) where:
#   ih_num = oh + pad - kh
#   iw_num = ow + pad - kw
#   if ih_num % stride == 0 and iw_num % stride == 0 and ih = ih_num/stride in [0,H_in) and iw in [0,W_in):
#       acc += input[n, ic, ih, iw] * weight[ic, oc, kh, kw]
# Fuse: out = tanh(acc + conv_bias[oc] - bias[oc])

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'OC', 'KH', 'KW', 'H_out', 'W_out'],
)
@triton.jit
def conv_transpose2d_fused_kernel(
    x_ptr,         # [N, IC, H_in, W_in]
    w_ptr,         # [IC, OC, KH, KW]
    cb_ptr,        # [OC]  conv bias
    bb_ptr,        # [OC]  user bias (broadcast)
    out_ptr,       # [N, OC, H_out, W_out]
    N, IC, OC,
    H_in, W_in,
    H_out, W_out,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,   # tile over spatial output (H_out*W_out)
    BLOCK_N: tl.constexpr,   # tile over OC
):
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)        # batch idx
    pid_oc = tl.program_id(2)       # oc tile

    # Spatial tile: M dimension is H_out*W_out
    M = H_out * W_out
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < OC

    oh = offs_m // W_out
    ow = offs_m % W_out

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # input batch offset
    x_batch_off = pid_n * IC * H_in * W_in

    # Loop over kernel positions and input channels
    for kh in tl.static_range(0, KH):
        ih_num = oh + PAD - kh
        ih = ih_num // STRIDE
        ih_valid = ((ih_num % STRIDE) == 0) & (ih >= 0) & (ih < H_in)
        for kw in tl.static_range(0, KW):
            iw_num = ow + PAD - kw
            iw = iw_num // STRIDE
            iw_valid = ((iw_num % STRIDE) == 0) & (iw >= 0) & (iw < W_in)
            spatial_valid = ih_valid & iw_valid  # [BLOCK_M]

            # input element offset (per row of M tile), once ic added
            in_sp_off = ih * W_in + iw  # [BLOCK_M]

            # weight offset within (ic, oc, kh, kw) plane: we need to iterate ic
            # weight[ic, oc, kh, kw] -> ic*OC*KH*KW + oc*KH*KW + kh*KW + kw
            w_khkw = kh * KW + kw

            for ic in range(0, IC):
                # load input: [BLOCK_M]
                x_off = x_batch_off + ic * H_in * W_in + in_sp_off
                x_vals = tl.load(
                    x_ptr + x_off,
                    mask=spatial_valid & m_mask,
                    other=0.0,
                )  # [BLOCK_M]

                # load weight slice: [BLOCK_N]
                w_off = ic * OC * KH * KW + offs_n * KH * KW + w_khkw
                w_vals = tl.load(
                    w_ptr + w_off,
                    mask=n_mask,
                    other=0.0,
                )  # [BLOCK_N]

                acc += x_vals[:, None] * w_vals[None, :]

    # add conv bias and subtract user bias, then tanh
    cb = tl.load(cb_ptr + offs_n, mask=n_mask, other=0.0)  # [BLOCK_N]
    bb = tl.load(bb_ptr + offs_n, mask=n_mask, other=0.0)  # [BLOCK_N]
    acc = acc + cb[None, :] - bb[None, :]

    # tanh
    # use closed form via exp; clamp for stability
    two_x = 2.0 * acc
    # tanh(x) = 1 - 2/(exp(2x)+1)
    e = tl.exp(two_x)
    out = 1.0 - 2.0 / (e + 1.0)

    # store
    out_batch_off = pid_n * OC * H_out * W_out
    out_off = out_batch_off + offs_n[None, :] * (H_out * W_out) + offs_m[:, None]
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_off, out, mask=store_mask)


def conv_transpose2d_fused(x, weight, conv_bias, user_bias, stride, padding, output_padding):
    N, IC, H_in, W_in = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w

    H_out = (H_in - 1) * stride - 2 * padding + KH + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + KW + output_padding

    out = torch.empty((N, OC, H_out, W_out), device=x.device, dtype=x.dtype)

    M = H_out * W_out

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            N,
            triton.cdiv(OC, meta['BLOCK_N']),
        )

    conv_transpose2d_fused_kernel[grid](
        x, weight, conv_bias, user_bias, out,
        N, IC, OC,
        H_in, W_in,
        H_out, W_out,
        KH, KW,
        stride, padding,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()
        cb = self.conv_transpose.bias.contiguous()
        bb = self.bias.view(-1).contiguous()
        out = conv_transpose2d_fused(
            x, w, cb, bb,
            self.stride, self.padding, self.output_padding,
        )
        return out