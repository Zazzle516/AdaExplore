import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_mean_kernel(
    x_ptr,           # (B, IC, D, H, W)
    w_ptr,           # (IC, OC, 3, 3, 3)  - ConvTranspose3d weight
    cb_ptr,          # (OC,) conv bias
    pb_ptr,          # (OC,) post bias (from self.bias)
    out_ptr,         # (B, OC, H, W) post-softmax-tanh-scale
    B, IC, D, H, W,
    scaling_factor,
    BLOCK_HW: tl.constexpr,
    OC: tl.constexpr,
):
    # one program per (b, hw_tile)
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)

    HW = H * W
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < HW
    h = offs_hw // W
    w = offs_hw % W

    # Accumulator: per (oc, hw) - sum over D of conv output
    # ConvTranspose3d with kernel=3, stride=1, padding=1 is equivalent to
    # Conv3d with weights flipped on spatial axes and weight layout (OC, IC, 3,3,3)
    # weight indexing: w[ic, oc, kd, kh, kw]
    # output[b,oc,d,h,w] = sum_{ic,kd,kh,kw} x[b,ic,d+1-kd, h+1-kh, w+1-kw] * w[ic,oc,kd,kh,kw]
    # for valid input indices.
    # We sum over d_out from 0..D-1 then divide by D.

    # accumulator shape (OC, BLOCK_HW)
    acc = tl.zeros((OC, BLOCK_HW), dtype=tl.float32)

    offs_oc = tl.arange(0, OC)

    # For sum over d_out: for each (kd), input d = d_out + 1 - kd
    # sum over d_out of x[b,ic, d_out+1-kd, ...] for d_out in [0, D)
    # = sum over d_in valid where d_in = d_out+1-kd, d_out in [0,D)
    # d_in ranges from 1-kd to D-kd. Valid d_in in [0, D).
    # So for kd=0: d_in in [1, D) -> contributes sum_{d_in=1..D-1} x[..d_in..]
    # for kd=1: d_in in [0, D)   -> full sum
    # for kd=2: d_in in [-1, D-1) -> d_in in [0, D-1) -> sum_{d_in=0..D-2}
    # i.e., sum_d_out x[b,ic,d_out+1-kd] = S - (boundary terms)
    # where S = sum_{d_in=0..D-1} x[b,ic,d_in,h_in,w_in]
    # kd=0: missing d_in=0
    # kd=1: full
    # kd=2: missing d_in=D-1

    # We materialize partial sums: full sum S[ic,h_in,w_in], plus x[:,:,0] and x[:,:,D-1]
    # Wait: that's a graph-level shortcut (pre-reducing along D). Forbidden by safety contract.

    # So we must actually do the full multiply-add. Loop over d_out and kd inside kernel.
    # Better: loop over d_in, and for each d_in determine which kd's hit it.
    # For each d_in in [0, D), it contributes to d_out = d_in + kd - 1, valid d_out in [0,D).
    # That's the same number of multiply-adds.
    # We accumulate the per-d_out sums then sum them. Equivalently:
    #   sum_{d_out} y[d_out] = sum_{d_in, kd valid} x[d_in] * w[kd] (for spatial pos)
    #                       = sum_{d_in} x[d_in] * sum_{kd: 0<=d_in+kd-1<D} w[kd]
    # But that pre-reduces kd over weights = also forbidden (folding kernel sum).

    # We must literally compute output[b,oc,d,h,w] for each d, then sum.
    # Loop over d_out, for each d_out compute conv output, accumulate.

    # Loop bounds
    for d_out in range(0, D):
        # for kd in 0..2: d_in = d_out + 1 - kd
        for kd in range(0, 3):
            d_in = d_out + 1 - kd
            if (d_in >= 0) and (d_in < D):
                for kh in range(0, 3):
                    h_in = h + 1 - kh  # vector
                    valid_h = (h_in >= 0) & (h_in < H)
                    for kw in range(0, 3):
                        w_in = w + 1 - kw
                        valid_w = (w_in >= 0) & (w_in < W)
                        valid_hw = valid_h & valid_w & mask_hw
                        # sum over ic
                        for ic in range(0, IC):
                            x_off = (((pid_b * IC + ic) * D + d_in) * H + h_in) * W + w_in
                            xv = tl.load(x_ptr + x_off, mask=valid_hw, other=0.0)  # (BLOCK_HW,)
                            # weight: w[ic, oc, kd, kh, kw], shape (IC, OC, 3,3,3)
                            w_off = ((((ic) * OC + offs_oc) * 3 + kd) * 3 + kh) * 3 + kw
                            wv = tl.load(w_ptr + w_off)  # (OC,)
                            # outer product add
                            acc += wv[:, None] * xv[None, :]

    # Add conv bias (per OC) * D (since accumulated over D d_outs)
    cb = tl.load(cb_ptr + offs_oc)  # (OC,)
    acc = acc + cb[:, None] * D

    # Divide by D for mean
    inv_D = 1.0 / D
    mean = acc * inv_D  # (OC, BLOCK_HW)

    # Add post bias
    pb = tl.load(pb_ptr + offs_oc)  # (OC,)
    mean = mean + pb[:, None]

    # Softmax across channels (axis 0)
    m_max = tl.max(mean, axis=0)  # (BLOCK_HW,)
    shifted = mean - m_max[None, :]
    e = tl.exp(shifted)
    s = tl.sum(e, axis=0)  # (BLOCK_HW,)
    sm = e / s[None, :]

    # tanh
    two_x = 2.0 * sm
    e2 = tl.exp(two_x)
    th = (e2 - 1.0) / (e2 + 1.0)
    out = th * scaling_factor

    # Store: output shape (B, OC, 1, H, W) - we write as (B, OC, H, W) flattened
    # out_ptr layout: (B, OC, H, W) with stride (OC*HW, HW, W, 1)
    out_off = (pid_b * OC + offs_oc[:, None]) * HW + offs_hw[None, :]
    tl.store(out_ptr + out_off, out, mask=mask_hw[None, :])


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Fast path: kernel=3, stride=1, padding=1
        if (self.kernel_size == 3 and self.stride == 1 and self.padding == 1
                and x.is_cuda and x.dtype == torch.float32):
            B, IC, D, H, W = x.shape
            OC = self.out_channels
            x = x.contiguous()
            weight = self.conv_transpose.weight.contiguous()  # (IC, OC, 3, 3, 3)
            cbias = self.conv_transpose.bias.contiguous() if self.conv_transpose.bias is not None \
                    else torch.zeros(OC, device=x.device, dtype=x.dtype)
            pbias = self.bias.view(-1).contiguous()

            out = torch.empty((B, OC, H, W), device=x.device, dtype=x.dtype)

            grid = lambda META: (B, (H * W + META['BLOCK_HW'] - 1) // META['BLOCK_HW'])
            fused_conv_mean_kernel[grid](
                x, weight, cbias, pbias, out,
                B, IC, D, H, W,
                float(self.scaling_factor),
                OC=OC,
                IC_C=IC,
            )
            return out.unsqueeze(2)

        # Fallback
        x = self.conv_transpose(x)
        x = x.mean(dim=2, keepdim=True)
        x = x + self.bias
        x = torch.softmax(x, dim=1)
        x = torch.tanh(x)
        x = x * self.scaling_factor
        return x