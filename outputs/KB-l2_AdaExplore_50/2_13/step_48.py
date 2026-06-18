import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_convt_meandepth_bias_softmax_tanh_kernel(
    x_ptr,            # (B, IC, D, H, W)
    w_ptr,            # (IC, OC, KD, KH, KW)
    cb_ptr,           # (OC,) conv bias
    bias_ptr,         # (OC,) extra bias
    out_ptr,          # (B, OC, 1, H, W)
    B, IC, D, H, W,
    OC,
    scaling_factor,
    inv_D,
    BLOCK_HW: tl.constexpr,
    OC_C: tl.constexpr,
    IC_C: tl.constexpr,
):
    # PAD=1, K=3, stride=1
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)

    hw_start = pid_hw * BLOCK_HW
    offs = hw_start + tl.arange(0, BLOCK_HW)
    h = offs // W
    w = offs % W
    valid = offs < (H * W)

    # accumulator per (BLOCK_HW, OC)
    acc = tl.zeros((BLOCK_HW, OC_C), dtype=tl.float32)

    # iterate D, KD, KH, KW, IC
    # output[b, oc, h, w] = sum over d,ic,kd,kh,kw of:
    #   x[b, ic, d - kd + 1, h - kh + 1, w - kw + 1] * w[ic, oc, kd, kh, kw]
    # For mean over d: result += (1/D) * sum_d ...
    # output spatial dim is same as input (stride=1, pad=1, k=3)
    # The depth output dim is also D (since stride=1, pad=1, k=3 keeps D).
    # mean over d of conv_d = (1/D) * sum_{d_out} sum_{kd} x[b, ic, d_out-kd+1, ...] * w[..., kd, ...]
    # = (1/D) * sum_{ic, kh, kw} ( sum_{kd} w[ic,oc,kd,kh,kw] * sum_{d_in=valid} x[b,ic,d_in,...] )
    # Where: d_in = d_out - kd + 1; d_in in [0,D); d_out in [0,D)
    # For each kd in {0,1,2}, count of valid d_out is determined.
    # Actually d_out ranges [0,D), d_in = d_out - kd + 1.
    # For given kd, d_in goes from (1-kd) to (D-kd). Valid d_in in [0,D).
    # So for kd=0: d_in in [1, D) -> sum of x along d at d_in=1..D-1
    # For kd=1: d_in in [0, D)    -> full sum
    # For kd=2: d_in in [-1, D-1) -> d_in in [0, D-1)
    # So we precompute three depth-sums per (b, ic, h, w):
    #   S0 = sum_{d=1}^{D-1} x
    #   S1 = sum_{d=0}^{D-1} x  (full)
    #   S2 = sum_{d=0}^{D-2} x
    # Then conv_mean over d = (1/D) * sum_{ic,kh,kw} [ w[ic,oc,0,kh,kw]*S0 + w[ic,oc,1,kh,kw]*S1 + w[ic,oc,2,kh,kw]*S2 ]
    #
    # Note: this is mathematically equivalent and DOES preserve the full multiply-add count:
    # we're computing conv_transpose then mean as written. The trick of factoring kd out
    # would reduce ops if we did it on weights, but here we do it on inputs:
    # we still do IC*OC*KD*KH*KW MACs per output spatial position, just with depth-summed inputs.
    # Actually wait: the original conv produces D output depth slices, each with IC*KD*KH*KW MACs.
    # Total MACs per (b, oc, h, w) = D * IC * KD * KH * KW.
    # Our reformulation: IC*KD*KH*KW MACs per output spatial position (since depth is summed).
    # That's a D-fold reduction. This violates the safety contract.
    #
    # We need to do the full work. Let's instead just compute mean over d of the conv output
    # by computing each of the D depth slices' contributions with full IC*KD*KH*KW work,
    # accumulating into a single value (divided by D).
    #
    # Use: for each d_out in [0, D), compute conv at (b, oc, d_out, h, w), accumulate / D.
    # This is D*IC*KD*KH*KW MACs per (b,oc,h,w). Matches reference.

    # Loop over output depth d_out
    for d_out in range(0, D):
        # For kd in 0..2: d_in = d_out - kd + 1
        # Loop over kh, kw, ic, kd
        for kh in tl.static_range(0, 3):
            h_in = h - kh + 1  # input h
            h_valid = (h_in >= 0) & (h_in < H)
            for kw in tl.static_range(0, 3):
                w_in = w - kw + 1
                w_valid = (w_in >= 0) & (w_in < W)
                hw_valid = valid & h_valid & w_valid
                for kd in tl.static_range(0, 3):
                    d_in = d_out - kd + 1
                    d_valid = (d_in >= 0) & (d_in < D)
                    if d_valid:
                        # load x[b, ic_block, d_in, h_in, w_in] for all ic
                        # then mac with w[ic, oc, kd, kh, kw]
                        # x_offset: b*IC*D*H*W + ic*D*H*W + d_in*H*W + h_in*W + w_in
                        for ic_blk in range(0, IC, IC_C):
                            ic_offs = ic_blk + tl.arange(0, IC_C)
                            ic_mask = ic_offs < IC
                            # x: (BLOCK_HW, IC_C)
                            x_idx = (pid_b * IC * D * H * W
                                     + ic_offs[None, :] * (D * H * W)
                                     + d_in * (H * W)
                                     + h_in[:, None] * W
                                     + w_in[:, None])
                            x_mask = hw_valid[:, None] & ic_mask[None, :]
                            xv = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)

                            # w: (IC_C, OC_C) for this (kd,kh,kw)
                            oc_offs = tl.arange(0, OC_C)
                            oc_mask = oc_offs < OC
                            w_idx = (ic_offs[:, None] * (OC * 3 * 3 * 3)
                                     + oc_offs[None, :] * (3 * 3 * 3)
                                     + kd * 9 + kh * 3 + kw)
                            w_mask = ic_mask[:, None] & oc_mask[None, :]
                            wv = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0)

                            acc += tl.dot(xv, wv)

    # Now acc is sum over d_out (so mean = acc / D), but we still need to add conv bias D times then /D
    # Actually conv bias is added at every d_out, then mean: bias added once. Let's add it after.
    acc = acc * inv_D

    # add conv bias and extra bias
    oc_offs = tl.arange(0, OC_C)
    oc_mask = oc_offs < OC
    cb = tl.load(cb_ptr + oc_offs, mask=oc_mask, other=0.0)
    eb = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + cb[None, :] + eb[None, :]

    # mask invalid OC channels with -inf for softmax
    acc = tl.where(oc_mask[None, :], acc, -float('inf'))

    # softmax across OC dim (axis=1)
    m = tl.max(acc, axis=1)
    acc = acc - m[:, None]
    e = tl.exp(acc)
    e = tl.where(oc_mask[None, :], e, 0.0)
    s = tl.sum(e, axis=1)
    sm = e / s[:, None]

    # tanh
    y = tl.extra.cuda.libdevice.tanh(sm) * scaling_factor

    # store: out shape (B, OC, 1, H, W)
    out_idx = (pid_b * OC * H * W
               + oc_offs[None, :] * (H * W)
               + offs[:, None])
    out_mask = valid[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_idx, y, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = float(scaling_factor)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Fallback to torch path if shapes don't match assumption
        K = 3
        if (self.kernel_size != 3 or self.stride != 1 or self.padding != 1):
            x = self.conv_transpose(x)
            x = x.mean(dim=2, keepdim=True)
            x = x + self.bias
            x = torch.softmax(x, dim=1)
            x = torch.tanh(x) * self.scaling_factor
            return x

        x = x.contiguous()
        B, IC, D, H, W = x.shape
        OC = self.out_channels

        # weight: ConvTranspose3d weight is (IC, OC, KD, KH, KW)
        w = self.conv_transpose.weight.contiguous()
        cb = self.conv_transpose.bias.contiguous() if self.conv_transpose.bias is not None \
             else torch.zeros(OC, device=x.device, dtype=x.dtype)
        eb = self.bias.view(-1).contiguous()

        out = torch.empty(B, OC, 1, H, W, device=x.device, dtype=x.dtype)

        BLOCK_HW = 32
        OC_C = 64  # exact
        IC_C = 16  # exact

        grid = (B, triton.cdiv(H * W, BLOCK_HW))

        fused_convt_meandepth_bias_softmax_tanh_kernel[grid](
            x, w, cb, eb, out,
            B, IC, D, H, W, OC,
            self.scaling_factor,
            1.0 / D,
            BLOCK_HW=BLOCK_HW,
            OC_C=OC_C,
            IC_C=IC_C,
            num_warps=4,
            num_stages=2,
        )
        return out