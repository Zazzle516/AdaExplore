import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 512}, num_warps=8, num_stages=2),
    ],
    key=['H', 'W'],
)
@triton.jit
def fused_conv_mean_kernel(
    x_ptr,           # (B, IC, D, H, W)
    w_ptr,           # (IC, OC, 3, 3, 3)  - ConvTranspose3d weight
    cb_ptr,          # (OC,) conv bias
    pb_ptr,          # (OC,) post bias (from self.bias)
    out_ptr,         # (B, OC, H, W) post-softmax-tanh-scale
    B, IC: tl.constexpr, D: tl.constexpr, H, W,
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

    acc = tl.zeros((OC, BLOCK_HW), dtype=tl.float32)

    offs_oc = tl.arange(0, OC)
    offs_ic = tl.arange(0, IC)

    # Swap loop order: (kd, kh, kw, d_out). Load weight tile once per (kd,kh,kw),
    # then reuse across all D d_out iterations.
    for kd in tl.static_range(0, 3):
        for kh in tl.static_range(0, 3):
            h_in = h + 1 - kh
            valid_h = (h_in >= 0) & (h_in < H)
            h_in_c = tl.where(valid_h, h_in, 0)
            for kw in tl.static_range(0, 3):
                w_in = w + 1 - kw
                valid_w = (w_in >= 0) & (w_in < W)
                w_in_c = tl.where(valid_w, w_in, 0)
                valid = valid_h & valid_w & mask_hw
                # weight slice (OC, IC) for current (kd,kh,kw) - loaded once
                w_off = (((offs_ic[None, :] * OC + offs_oc[:, None]) * 3 + kd) * 3 + kh) * 3 + kw
                wv = tl.load(w_ptr + w_off)
                # Loop over d_out, accumulating using same weight tile
                for d_out in range(0, D):
                    d_in = d_out + 1 - kd
                    if (d_in >= 0) and (d_in < D):
                        x_off = (((pid_b * IC + offs_ic[:, None]) * D + d_in) * H
                                 + h_in_c[None, :]) * W + w_in_c[None, :]
                        xv = tl.load(x_ptr + x_off, mask=valid[None, :], other=0.0)
                        acc += tl.dot(wv, xv, allow_tf32=True)

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

            grid = lambda meta: (B, (H * W + meta['BLOCK_HW'] - 1) // meta['BLOCK_HW'])
            fused_conv_mean_kernel[grid](
                x, weight, cbias, pbias, out,
                B, IC, D, H, W,
                float(self.scaling_factor),
                OC=OC,
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