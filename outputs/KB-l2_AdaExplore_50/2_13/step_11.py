import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _conv_mean_fused_kernel(
    x_ptr,        # (B, IC, D, H, W)
    w_ptr,        # (IC, OC, KD, KH, KW)
    cb_ptr,       # (OC,)
    bias_ptr,     # (OC,)
    out_ptr,      # (B, OC, H, W)
    B, IC, D, H, W,
    OC,
    scaling_factor,
    inv_D,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    IC_C: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    PAD: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    HW = H * W

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    mask_oc = offs_oc < OC
    mask_hw = offs_hw < HW

    h_out = offs_hw // W
    w_out = offs_hw % W

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    offs_ic = tl.arange(0, IC_C)
    mask_ic = offs_ic < IC

    for kd in tl.static_range(0, KD):
        d_out_lo = tl.maximum(0, kd - PAD)
        d_out_hi = tl.minimum(D - 1, D - 1 + kd - PAD)
        d_in_lo = d_out_lo - kd + PAD
        d_in_hi = d_out_hi - kd + PAD

        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                h_in = h_out - kh + PAD
                w_in = w_out - kw + PAD
                spatial_valid = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W) & mask_hw

                w_offs = (offs_ic[:, None] * (OC * KD * KH * KW)
                          + offs_oc[None, :] * (KD * KH * KW)
                          + kd * (KH * KW) + kh * KW + kw)
                w_mask = mask_ic[:, None] & mask_oc[None, :]
                w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

                x_sum = tl.zeros((IC_C, BLOCK_HW), dtype=tl.float32)

                for d_in in range(d_in_lo, d_in_hi + 1):
                    x_offs = (pid_b * (IC * D * H * W)
                              + offs_ic[:, None] * (D * H * W)
                              + d_in * (H * W)
                              + h_in[None, :] * W + w_in[None, :])
                    x_mask = mask_ic[:, None] & spatial_valid[None, :]
                    x_vals = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)
                    x_sum += x_vals

                acc += tl.dot(tl.trans(w_vals), x_sum)

    acc = acc * inv_D

    cb = tl.load(cb_ptr + offs_oc, mask=mask_oc, other=0.0)
    eb = tl.load(bias_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + (cb + eb)[:, None]

    out_offs = (pid_b * (OC * HW)
                + offs_oc[:, None] * HW
                + offs_hw[None, :])
    out_mask = mask_oc[:, None] & mask_hw[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


@triton.jit
def _softmax_tanh_scale_kernel(
    x_ptr,
    out_ptr,
    B, C, H, W,
    scaling_factor,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)

    HW = H * W

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < HW

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    # Load full (C, BLOCK_HW) tile
    x_ptrs = x_ptr + pid_b * C * HW + offs_c[:, None] * HW + offs_hw[None, :]
    mask = mask_c[:, None] & mask_hw[None, :]
    v = tl.load(x_ptrs, mask=mask, other=-float('inf'))

    m = tl.max(v, axis=0)  # [BLOCK_HW]
    e = tl.exp(v - m[None, :])
    e = tl.where(mask_c[:, None], e, 0.0)
    s = tl.sum(e, axis=0)  # [BLOCK_HW]
    sm = e / s[None, :]

    e2 = tl.exp(2.0 * sm)
    t = (e2 - 1.0) / (e2 + 1.0)
    out = t * scaling_factor

    out_ptrs = out_ptr + pid_b * C * HW + offs_c[:, None] * HW + offs_hw[None, :]
    tl.store(out_ptrs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scaling_factor = scaling_factor

        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))

    def forward(self, x):
        B, IC, D, H, W = x.shape
        OC = self.out_channels
        K = self.kernel_size
        PAD = self.padding

        x = x.contiguous()
        weight = self.conv_transpose.weight.contiguous()
        cb = self.conv_transpose.bias.contiguous() if self.conv_transpose.bias is not None else torch.zeros(OC, device=x.device, dtype=x.dtype)
        bias_flat = self.bias.view(-1).contiguous()

        reduced = torch.empty((B, OC, H, W), device=x.device, dtype=x.dtype)

        BLOCK_OC = 64
        BLOCK_HW = 64
        ic_c = 1
        while ic_c < IC:
            ic_c *= 2
        IC_C = ic_c

        grid = (B, triton.cdiv(OC, BLOCK_OC), triton.cdiv(H * W, BLOCK_HW))

        _conv_mean_fused_kernel[grid](
            x, weight, cb, bias_flat, reduced,
            B, IC, D, H, W, OC,
            float(self.scaling_factor),
            1.0 / float(D),
            BLOCK_OC=BLOCK_OC,
            BLOCK_HW=BLOCK_HW,
            IC_C=IC_C,
            KD=K, KH=K, KW=K,
            PAD=PAD,
            num_warps=4,
            num_stages=2,
        )

        out = torch.empty_like(reduced)
        BLOCK_C = 1
        while BLOCK_C < OC:
            BLOCK_C *= 2
        BLOCK_HW2 = 64

        grid2 = (B, triton.cdiv(H * W, BLOCK_HW2))
        _softmax_tanh_scale_kernel[grid2](
            reduced, out,
            B, OC, H, W,
            float(self.scaling_factor),
            BLOCK_C=BLOCK_C,
            BLOCK_HW=BLOCK_HW2,
            num_warps=4,
        )

        return out.unsqueeze(2)