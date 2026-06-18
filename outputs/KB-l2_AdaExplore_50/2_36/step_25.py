import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose_min_sum_gelu_bias_kernel(
    x_ptr,        # (N, IC, H_in, W_in)
    w_ptr,        # (IC, OC, KH, KW)
    cb_ptr,       # (OC,) conv bias
    bias_ptr,     # scalar
    out_ptr,      # (N, 1, 1, W_out)
    N, IC, H_in, W_in, OC, H_out, W_out,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, w_out)
    pid = tl.program_id(0)
    n = pid // W_out
    w_out = pid % W_out

    offs_oc = tl.arange(0, BLOCK_OC)
    oc_mask = offs_oc < OC

    # accumulator for sum over h_out of min over oc
    sum_acc = 0.0

    # input base for this n
    x_n_base = n * IC * H_in * W_in

    # iterate over h_out
    for h_out in range(0, H_out):
        # min across oc for this (n, h_out, w_out)
        # val[oc] = conv_bias[oc] + sum_{ic, kh, kw} x[n, ic, h_in, w_in] * w[ic, oc, kh, kw]
        # where h_in = (h_out + PAD - kh) / STRIDE if divisible, similarly for w
        # Initialize accumulator vector [BLOCK_OC]
        cb = tl.load(cb_ptr + offs_oc, mask=oc_mask, other=0.0)
        acc = cb

        # loop over kh, kw
        for kh in range(0, KH):
            h_num = h_out + PAD - kh
            h_in = h_num // STRIDE
            h_valid = (h_num >= 0) & (h_num - h_in * STRIDE == 0) & (h_in >= 0) & (h_in < H_in)
            for kw in range(0, KW):
                w_num = w_out + PAD - kw
                w_in = w_num // STRIDE
                w_valid = (w_num >= 0) & (w_num - w_in * STRIDE == 0) & (w_in >= 0) & (w_in < W_in)
                valid = h_valid & w_valid
                if valid:
                    # accumulate over ic
                    x_base = x_n_base + h_in * W_in + w_in  # add ic * H_in*W_in
                    w_base = (kh * KW + kw) * OC + offs_oc  # weight shape (IC, OC, KH, KW) -> (IC, KH*KW*OC)? we'll reorder
                    # We'll layout weight as (IC, KH, KW, OC) for contiguous OC loads
                    # weight index: ic*KH*KW*OC + kh*KW*OC + kw*OC + oc
                    for ic in range(0, IC):
                        x_val = tl.load(x_ptr + x_base + ic * H_in * W_in)
                        w_vals = tl.load(
                            w_ptr + ic * KH * KW * OC + kh * KW * OC + kw * OC + offs_oc,
                            mask=oc_mask, other=0.0,
                        )
                        acc = acc + x_val * w_vals

        # apply mask: invalid oc -> +inf
        acc = tl.where(oc_mask, acc, float('inf'))
        m = tl.min(acc, axis=0)
        sum_acc = sum_acc + m

    # GELU exact: 0.5 * s * (1 + erf(s/sqrt(2)))
    inv_sqrt2 = 0.70710678118654752440
    g = 0.5 * sum_acc * (1.0 + tl.erf(sum_acc * inv_sqrt2))
    b = tl.load(bias_ptr)
    out = g + b
    tl.store(out_ptr + n * W_out + w_out, out)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        # Pre-permute weight to (IC, KH, KW, OC) contiguous for fast loads
        # nn.ConvTranspose2d weight shape: (IC, OC, KH, KW)
        self._weight_cache = None
        self._bias_cache = None

    def _get_weight(self):
        w = self.conv_transpose.weight  # (IC, OC, KH, KW)
        if (self._weight_cache is None or
                self._weight_cache.shape[0] != w.shape[0] or
                self._weight_cache.device != w.device or
                self._weight_cache.data_ptr() == 0):
            wp = w.detach().permute(0, 2, 3, 1).contiguous()  # (IC, KH, KW, OC)
            self._weight_cache = wp
            cb = self.conv_transpose.bias
            if cb is None:
                self._bias_cache = torch.zeros(w.shape[1], device=w.device, dtype=w.dtype)
            else:
                self._bias_cache = cb.detach().contiguous()
        return self._weight_cache, self._bias_cache

    def forward(self, x):
        x = x.contiguous()
        N, IC, H_in, W_in = x.shape
        OC = self.out_channels
        KH = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0]
        KW = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[1]
        STRIDE = self.stride if isinstance(self.stride, int) else self.stride[0]
        PAD = self.padding if isinstance(self.padding, int) else self.padding[0]
        OP = self.output_padding if isinstance(self.output_padding, int) else self.output_padding[0]

        H_out = (H_in - 1) * STRIDE - 2 * PAD + KH + OP
        W_out = (W_in - 1) * STRIDE - 2 * PAD + KW + OP

        w_perm, cb = self._get_weight()

        out = torch.empty((N, 1, 1, W_out), device=x.device, dtype=x.dtype)

        # BLOCK_OC must be >= OC and power of 2
        BLOCK_OC = 1
        while BLOCK_OC < OC:
            BLOCK_OC *= 2

        grid = (N * W_out,)
        conv_transpose_min_sum_gelu_bias_kernel[grid](
            x, w_perm, cb, self.bias, out,
            N, IC, H_in, W_in, OC, H_out, W_out,
            KH=KH, KW=KW, STRIDE=STRIDE, PAD=PAD,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
        )
        return out