import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_mean_kernel(
    x_ptr,        # (B, IC, D, H, W)
    w_ptr,        # (IC, OC, kD, kH, kW)
    cb_ptr,       # (OC,) conv bias
    bias_ptr,     # (OC,) extra bias
    out_ptr,      # (B, OC, H, W)
    B, IC, D, H, W,
    OC,
    scaling_factor,
    BLOCK_HW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < (H * W)
    h_idx = offs_hw // W
    w_idx = offs_hw % W

    offs_oc = tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    # Accumulate: sum over D, kD, kH, kW, IC of x[b,ic,d-1+kd? actually conv_transpose]
    # ConvTranspose3d with stride=1, padding=1, kernel=3 is equivalent to Conv3d with same padding=1
    # (since stride=1 and weight is just flipped). For mean over D we sum over D and divide.
    # Actually: ConvTranspose3d(stride=1, padding=p) computes:
    #   y[n,oc,d,h,w] = sum_{ic,kd,kh,kw} x[n,ic,d+kd-p_eff, ...] * w[ic,oc,kd,kh,kw]
    # where output shape = (D - 1)*stride - 2*padding + kernel = D for s=1,p=1,k=3.
    # The relation: y[d,h,w] = sum_{ic,kd,kh,kw} x[ic, d - kd + (k-1) - p, ...] * w[ic,oc,kd,kh,kw]
    # For k=3, p=1, k-1-p = 1. So input index for kd is d - kd + 1.
    # We sum y over d=0..D-1. For each (kd, ih), the input index ih = d - kd + 1 ranges over
    # valid d. Total contribution = sum over valid d of x[ic, ih, ...].
    # So sum_d y[d] = sum_{ic,kd,kh,kw,d} x[ic, d-kd+1, ...] * w[...]
    # But we MUST keep the full compute count per safety contract. So we explicitly loop.

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    kD = 3
    kH = 3
    kW = 3
    pad = 1

    # Reorder: for each (ic, kh, kw), pre-load 3 weights (one per kd), then loop
    # over d_in and reuse the same x across the 3 kd accumulations.
    for ic in range(0, IC):
        for kh in range(0, kH):
            h_in = h_idx + pad - kh
            mask_h = (h_in >= 0) & (h_in < H)
            for kw_ in range(0, kW):
                w_in = w_idx + pad - kw_
                mask_w = (w_in >= 0) & (w_in < W)
                spatial_mask = mask_hw & mask_h & mask_w

                # Load weights for kd = 0, 1, 2 (vector over OC)
                w_base = (ic * OC * kD * kH * kW
                          + offs_oc * kD * kH * kW
                          + kh * kW
                          + kw_)
                w0 = tl.load(w_ptr + w_base + 0 * kH * kW, mask=mask_oc, other=0.0)
                w1 = tl.load(w_ptr + w_base + 1 * kH * kW, mask=mask_oc, other=0.0)
                w2 = tl.load(w_ptr + w_base + 2 * kH * kW, mask=mask_oc, other=0.0)

                x_row_base = (pid_b * IC * D * H * W
                              + ic * D * H * W
                              + h_in * W
                              + w_in)

                for d_in in range(0, D):
                    x_off = x_row_base + d_in * H * W
                    xv = tl.load(x_ptr + x_off, mask=spatial_mask, other=0.0)
                    # kd=0: output d = d_in - 1, valid if d_in >= 1
                    # kd=1: output d = d_in, always valid for d_in in [0,D)
                    # kd=2: output d = d_in + 1, valid if d_in <= D-2
                    if d_in >= 1:
                        acc += w0[:, None] * xv[None, :]
                    acc += w1[:, None] * xv[None, :]
                    if d_in <= D - 2:
                        acc += w2[:, None] * xv[None, :]

    # Add conv bias * D (since bias is added per output position; summing over D multiplies by D)
    cb = tl.load(cb_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + (cb * D)[:, None]

    # Mean over D
    acc = acc / D

    # Add extra bias
    extra_b = tl.load(bias_ptr + offs_oc, mask=mask_oc, other=0.0)
    v = acc + extra_b[:, None]

    # Softmax over OC
    v_safe = tl.where(mask_oc[:, None], v, -float('inf'))
    m = tl.max(v_safe, axis=0)
    e = tl.exp(v - m[None, :])
    e = tl.where(mask_oc[:, None], e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s[None, :]

    # tanh
    t = 2.0 * tl.sigmoid(2.0 * sm) - 1.0
    out = t * scaling_factor

    # Write output
    out_off = (pid_b * OC * H * W
               + offs_oc[:, None] * H * W
               + offs_hw[None, :])
    mask_store = mask_oc[:, None] & mask_hw[None, :]
    tl.store(out_ptr + out_off, out, mask=mask_store)


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
        # Fallback to torch if shapes don't match expectations
        if (self.kernel_size != 3 or self.stride != 1 or self.padding != 1):
            x = self.conv_transpose(x)
            x = x.mean(dim=2, keepdim=True)
            x = x + self.bias
            x = torch.softmax(x, dim=1)
            x = torch.tanh(x)
            x = x * self.scaling_factor
            return x

        x = x.contiguous()
        B, IC, D, H, W = x.shape
        OC = self.out_channels

        # weight shape: (IC, OC, kD, kH, kW)
        w = self.conv_transpose.weight.contiguous()
        cb = self.conv_transpose.bias.contiguous() if self.conv_transpose.bias is not None else torch.zeros(OC, device=x.device, dtype=x.dtype)
        bias_flat = self.bias.view(-1).contiguous()

        out = torch.empty((B, OC, H, W), device=x.device, dtype=x.dtype)

        BLOCK_HW = 64
        BLOCK_OC = triton.next_power_of_2(OC)

        grid = (B, triton.cdiv(H * W, BLOCK_HW))
        fused_conv_mean_kernel[grid](
            x, w, cb, bias_flat, out,
            B, IC, D, H, W, OC,
            float(self.scaling_factor),
            BLOCK_HW=BLOCK_HW,
            BLOCK_OC=BLOCK_OC,
            num_warps=8,
            num_stages=2,
        )
        return out.view(B, OC, 1, H, W)