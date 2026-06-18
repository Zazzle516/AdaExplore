import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_fused_kernel(
    x_ptr,           # input (N, IC, D, H, W)
    w_ptr,           # weight (OC, IC, KT, KH, KW)
    cb_ptr,          # conv bias (OC,)
    scale_ptr,       # scaling factor (OC,)
    bias_ptr,        # bias (OC,)
    out_ptr,         # output (N, OC, D_out, H_out, W_out)
    N, D_in, H_in, W_in,
    D_out, H_out, W_out,
    IC: tl.constexpr,
    OC: tl.constexpr,
    KT: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # Grid: (HW tile index, D_out, N)
    pid_hw = tl.program_id(0)
    pid_d  = tl.program_id(1)
    pid_n  = tl.program_id(2)

    HW_out = H_out * W_out
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offs < HW_out

    h_idx = hw_offs // W_out
    w_idx = hw_offs % W_out
    d_idx = pid_d

    oc_offs = tl.arange(0, OC)  # (OC,)

    # Preload weights: shape (OC, IC*KT*KH*KW) flat
    # weight layout (OC, IC, KT, KH, KW) -> index oc * IC*KT*KH*KW + ic*KT*KH*KW + kt*KH*KW + kh*KW + kw
    K_total = IC * KT * KH * KW

    # Load conv bias once
    cb = tl.load(cb_ptr + oc_offs)        # (OC,)
    sc = tl.load(scale_ptr + oc_offs)     # (OC,)
    bs = tl.load(bias_ptr + oc_offs)      # (OC,)

    acc = tl.zeros((BLOCK_HW, OC), dtype=tl.float32)

    # input batch pointer
    x_batch_ptr = x_ptr + pid_n * IC * D_in * H_in * W_in

    # Loop over IC, KT, KH, KW (small: 3*3*3*3 = 81)
    for ic in tl.static_range(IC):
        for kt in tl.static_range(KT):
            in_d = d_idx + kt
            for kh in tl.static_range(KH):
                in_h = h_idx + kh  # (BLOCK_HW,)
                for kw in tl.static_range(KW):
                    in_w = w_idx + kw  # (BLOCK_HW,)
                    # input offset for this (ic, in_d, in_h, in_w)
                    x_off = ic * (D_in * H_in * W_in) + in_d * (H_in * W_in) + in_h * W_in + in_w
                    x_vals = tl.load(x_batch_ptr + x_off, mask=hw_mask, other=0.0)  # (BLOCK_HW,)

                    # weight offset for all OC at (ic, kt, kh, kw)
                    w_off = oc_offs * K_total + ic * (KT * KH * KW) + kt * (KH * KW) + kh * KW + kw
                    w_vals = tl.load(w_ptr + w_off)  # (OC,)

                    acc += x_vals[:, None] * w_vals[None, :]

    # Epilogue
    v = (acc + cb[None, :]) * sc[None, :]
    e = tl.exp(2.0 * v)
    t = (e - 1.0) / (e + 1.0)
    y = t * bs[None, :]
    out = 1.0 / (1.0 + tl.exp(-y))

    # Store output (N, OC, D_out, H_out, W_out)
    DHW = D_out * H_out * W_out
    out_batch_ptr = out_ptr + pid_n * OC * DHW
    out_off = oc_offs[None, :] * DHW + pid_d * HW_out + hw_offs[:, None]
    tl.store(out_batch_ptr + out_off, out, mask=hw_mask[:, None])


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, IC, D_in, H_in, W_in = x.shape
        OC = self.out_channels
        KT = KH = KW = self.kernel_size
        D_out = D_in - KT + 1
        H_out = H_in - KH + 1
        W_out = W_in - KW + 1

        w = self.conv.weight.contiguous()
        cb = self.conv.bias.contiguous()
        scale = self.scaling_factor.contiguous().view(-1)
        bias = self.bias.contiguous().view(-1)

        out = torch.empty((N, OC, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

        HW_out = H_out * W_out
        BLOCK_HW = 128
        grid = ((HW_out + BLOCK_HW - 1) // BLOCK_HW, D_out, N)

        conv3d_fused_kernel[grid](
            x, w, cb, scale, bias, out,
            N, D_in, H_in, W_in,
            D_out, H_out, W_out,
            IC=IC, OC=OC, KT=KT, KH=KH, KW=KW,
            BLOCK_HW=BLOCK_HW,
            num_warps=4,
            num_stages=2,
        )
        return out