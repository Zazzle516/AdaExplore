import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose_min_kernel(
    x_ptr,        # (N, IC, H_in, W_in)
    w_ptr,        # (KH, KW, IC, OC) contiguous
    cb_ptr,       # (OC,)
    out_ptr,      # (N, H_out, W_out)  -- min over OC
    N, IC, H_in, W_in, OC, H_out, W_out,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # program per (n, h_out, w_out_tile)
    pid = tl.program_id(0)
    pid_w = tl.program_id(1)
    n = pid // H_out
    h_out = pid % H_out

    w_start = pid_w * BLOCK_W
    offs_w = w_start + tl.arange(0, BLOCK_W)
    w_mask = offs_w < W_out

    offs_oc = tl.arange(0, BLOCK_OC)
    oc_mask = offs_oc < OC

    # accumulator [BLOCK_W, BLOCK_OC]
    cb = tl.load(cb_ptr + offs_oc, mask=oc_mask, other=0.0)
    acc = tl.broadcast_to(cb[None, :], (BLOCK_W, BLOCK_OC))

    x_n_base = n * IC * H_in * W_in

    for kh in tl.static_range(0, KH):
        h_num = h_out + PAD - kh
        h_in = h_num // STRIDE
        h_valid = (h_num >= 0) & (h_num - h_in * STRIDE == 0) & (h_in >= 0) & (h_in < H_in)
        for kw in tl.static_range(0, KW):
            w_num = offs_w + PAD - kw
            w_in = w_num // STRIDE
            w_valid = (w_num >= 0) & (w_num - w_in * STRIDE == 0) & (w_in >= 0) & (w_in < W_in) & w_mask
            valid = w_valid & h_valid  # [BLOCK_W]

            # iterate IC in blocks
            for ic_start in range(0, IC, BLOCK_IC):
                offs_ic = ic_start + tl.arange(0, BLOCK_IC)
                ic_mask = offs_ic < IC

                # load x[n, ic, h_in, w_in] -> [BLOCK_W, BLOCK_IC]
                x_offs = x_n_base + offs_ic[None, :] * (H_in * W_in) + h_in * W_in + w_in[:, None]
                x_load_mask = valid[:, None] & ic_mask[None, :]
                x_vals = tl.load(x_ptr + x_offs, mask=x_load_mask, other=0.0)

                # load w[kh, kw, ic, oc] -> [BLOCK_IC, BLOCK_OC]
                w_offs = (kh * KW + kw) * (IC * OC) + offs_ic[:, None] * OC + offs_oc[None, :]
                w_load_mask = ic_mask[:, None] & oc_mask[None, :]
                w_vals = tl.load(w_ptr + w_offs, mask=w_load_mask, other=0.0)

                acc = acc + tl.dot(x_vals, w_vals)

    # mask invalid oc to +inf, then min along oc
    acc = tl.where(oc_mask[None, :], acc, float('inf'))
    m = tl.min(acc, axis=1)  # [BLOCK_W]

    # store
    out_base = n * H_out * W_out + h_out * W_out
    tl.store(out_ptr + out_base + offs_w, m, mask=w_mask)


@triton.jit
def sum_gelu_bias_kernel(
    x_ptr,      # (N, H_out, W_out)
    bias_ptr,   # scalar
    out_ptr,    # (N, 1, 1, W_out)
    N, H, W,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_w = tl.program_id(1)
    n = pid
    w_start = pid_w * BLOCK_W
    offs_w = w_start + tl.arange(0, BLOCK_W)
    w_mask = offs_w < W

    offs_h = tl.arange(0, BLOCK_H)
    h_mask = offs_h < H

    # load (BLOCK_H, BLOCK_W)
    x_offs = n * H * W + offs_h[:, None] * W + offs_w[None, :]
    mask = h_mask[:, None] & w_mask[None, :]
    x = tl.load(x_ptr + x_offs, mask=mask, other=0.0)
    s = tl.sum(x, axis=0)  # [BLOCK_W]

    inv_sqrt2 = 0.70710678118654752440
    g = 0.5 * s * (1.0 + tl.erf(s * inv_sqrt2))
    b = tl.load(bias_ptr)
    out = g + b
    tl.store(out_ptr + n * W + offs_w, out, mask=w_mask)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


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
        self._weight_cache = None
        self._cb_cache = None
        self._weight_ver = None

    def _get_weight(self):
        w = self.conv_transpose.weight  # (IC, OC, KH, KW)
        ver = w._version
        if self._weight_cache is None or self._weight_ver != ver or self._weight_cache.device != w.device:
            # permute to (KH, KW, IC, OC) contiguous
            wp = w.detach().permute(2, 3, 0, 1).contiguous()
            self._weight_cache = wp
            cb = self.conv_transpose.bias
            if cb is None:
                self._cb_cache = torch.zeros(w.shape[1], device=w.device, dtype=w.dtype)
            else:
                self._cb_cache = cb.detach().contiguous()
            self._weight_ver = ver
        return self._weight_cache, self._cb_cache

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

        BLOCK_OC = _next_pow2(OC)
        BLOCK_IC = 32 if IC >= 32 else _next_pow2(IC)
        BLOCK_W = 64

        min_out = torch.empty((N, H_out, W_out), device=x.device, dtype=x.dtype)
        grid1 = (N * H_out, (W_out + BLOCK_W - 1) // BLOCK_W)
        conv_transpose_min_kernel[grid1](
            x, w_perm, cb, min_out,
            N, IC, H_in, W_in, OC, H_out, W_out,
            KH=KH, KW=KW, STRIDE=STRIDE, PAD=PAD,
            BLOCK_OC=BLOCK_OC, BLOCK_IC=BLOCK_IC, BLOCK_W=BLOCK_W,
            num_warps=4, num_stages=2,
        )

        out = torch.empty((N, 1, 1, W_out), device=x.device, dtype=x.dtype)
        BLOCK_H2 = _next_pow2(H_out)
        BLOCK_W2 = 64
        grid2 = (N, (W_out + BLOCK_W2 - 1) // BLOCK_W2)
        sum_gelu_bias_kernel[grid2](
            min_out, self.bias, out,
            N, H_out, W_out,
            BLOCK_H=BLOCK_H2, BLOCK_W=BLOCK_W2,
            num_warps=4,
        )
        return out