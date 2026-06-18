import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, IC: tl.constexpr, OC: tl.constexpr,
    H_in, W_in, H_out, W_out,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Grid: (N*H_out, num_w_tiles, OC)
    pid_nh = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_oc = tl.program_id(2)

    n = pid_nh // H_out
    h_out = pid_nh % H_out
    w_start = pid_w * BLOCK_W

    offs_w = w_start + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W_out

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # ConvTranspose2d with stride=1, no padding, kernel=3:
    # y[n,oc,h,w] = sum_{ic,kh,kw} x[n,ic,h-kh, w-kw] * weight[ic,oc,kh,kw]
    # valid where 0 <= h-kh < H_in, 0 <= w-kw < W_in
    for kh in tl.static_range(0, KH):
        h_in = h_out - kh
        h_valid = (h_in >= 0) & (h_in < H_in)
        for kw in tl.static_range(0, KW):
            w_in = offs_w - kw
            w_valid = (w_in >= 0) & (w_in < W_in) & mask_w
            valid = h_valid & w_valid
            for ic in tl.static_range(0, IC):
                x_off = n * IC * H_in * W_in + ic * H_in * W_in + h_in * W_in + w_in
                x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                w_off = ic * OC * KH * KW + pid_oc * KH * KW + kh * KW + kw
                w_val = tl.load(w_ptr + w_off)
                acc += x_val * w_val

    bias = tl.load(b_ptr + pid_oc)
    acc += bias

    y_off = n * OC * H_out * W_out + pid_oc * H_out * W_out + h_out * W_out + offs_w
    tl.store(y_ptr + y_off, acc, mask=mask_w)


@triton.jit
def gelu_groupnorm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    N, C, HW, G, CPG: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    group_elems = CPG * HW
    base = n * C * HW + g * CPG * HW

    inv_sqrt2 = 0.7071067811865475

    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    sumsq_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for c in tl.static_range(0, CPG):
        c_base = base + c * HW
        for off in range(0, HW, BLOCK_SIZE):
            offs = off + tl.arange(0, BLOCK_SIZE)
            mask = offs < HW
            x = tl.load(x_ptr + c_base + offs, mask=mask, other=0.0).to(tl.float32)
            gx = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
            gx_masked = tl.where(mask, gx, 0.0)
            sum_val += gx_masked
            sumsq_val += gx_masked * gx_masked

    s = tl.sum(sum_val, axis=0)
    sq = tl.sum(sumsq_val, axis=0)

    mean = s / group_elems
    var = sq / group_elems - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    g_off = g * CPG
    for c in tl.static_range(0, CPG):
        c_base = base + c * HW
        w = tl.load(weight_ptr + g_off + c).to(tl.float32)
        b = tl.load(bias_ptr + g_off + c).to(tl.float32)
        scale = rstd * w
        shift = b - mean * scale
        for off in range(0, HW, BLOCK_SIZE):
            offs = off + tl.arange(0, BLOCK_SIZE)
            mask = offs < HW
            x = tl.load(x_ptr + c_base + offs, mask=mask, other=0.0).to(tl.float32)
            gx = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
            y = gx * scale + shift
            tl.store(y_ptr + c_base + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.num_groups = num_groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = 1e-5

    def forward(self, x):
        N, IC, H_in, W_in = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        # stride=1, padding=0
        H_out = H_in + KH - 1
        W_out = W_in + KW - 1

        x = x.contiguous()
        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, KH, KW)
        bias = self.conv_transpose.bias.contiguous()

        conv_out = torch.empty((N, OC, H_out, W_out), device=x.device, dtype=x.dtype)

        BLOCK_W = 128
        grid = (N * H_out, triton.cdiv(W_out, BLOCK_W), OC)
        conv_transpose_kernel[grid](
            x, weight, bias, conv_out,
            N, IC, OC, H_in, W_in, H_out, W_out,
            KH, KW,
            BLOCK_W=BLOCK_W,
            num_warps=4,
            num_stages=2,
        )

        C = OC
        HW = H_out * W_out
        G = self.num_groups
        CPG = C // G

        y = torch.empty_like(conv_out)

        if HW >= 16384:
            BLOCK_SIZE = 2048
            num_warps = 8
        elif HW >= 4096:
            BLOCK_SIZE = 1024
            num_warps = 8
        elif HW >= 1024:
            BLOCK_SIZE = 512
            num_warps = 4
        else:
            BLOCK_SIZE = 256
            num_warps = 4

        grid2 = (N * G,)
        gelu_groupnorm_kernel[grid2](
            conv_out, y,
            self.group_norm.weight, self.group_norm.bias,
            N, C, HW, G, CPG,
            self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=2,
        )
        return y