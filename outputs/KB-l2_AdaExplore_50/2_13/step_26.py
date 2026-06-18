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

    # Loop over d (output), kd, kh, kw, ic
    # Input shape: x[b, ic, d_in, h_in, w_in], d_in = d - kd + 1, etc for h,w
    # For mean over D: divide by D in epilogue
    kD = 3
    kH = 3
    kW = 3
    pad = 1

    for d in range(0, D):
        for kd in range(0, kD):
            d_in = d - kd + pad  # since output_pad? for ConvT: input idx = (d + pad - kd) when stride=1
            # ConvTranspose with stride=1: out[d] = sum_{kd} in[d+pad-kd] * w[kd]
            d_valid = (d_in >= 0) & (d_in < D)
            if d_valid:
                for kh in range(0, kH):
                    for kw_ in range(0, kW):
                        h_in = h_idx + pad - kh
                        w_in = w_idx + pad - kw_
                        mask_h = (h_in >= 0) & (h_in < H)
                        mask_w = (w_in >= 0) & (w_in < W)
                        spatial_mask = mask_hw & mask_h & mask_w

                        for ic in range(0, IC):
                            # Load input element x[b, ic, d_in, h_in, w_in]
                            x_off = (pid_b * IC * D * H * W
                                     + ic * D * H * W
                                     + d_in * H * W
                                     + h_in * W
                                     + w_in)
                            xv = tl.load(x_ptr + x_off, mask=spatial_mask, other=0.0)
                            # Load weight w[ic, oc, kd, kh, kw]
                            w_off = (ic * OC * kD * kH * kW
                                     + offs_oc * kD * kH * kW
                                     + kd * kH * kW
                                     + kh * kW
                                     + kw_)
                            wv = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)
                            acc += wv[:, None] * xv[None, :]

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

        BLOCK_HW = 32
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