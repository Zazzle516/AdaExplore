import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_mean_invstd_kernel(
    x_ptr, w_ptr, b_ptr, mult_ptr,
    conv_out_ptr, mean_ptr, invstd_ptr,
    N, C_IN: tl.constexpr, D_IN: tl.constexpr, H_IN: tl.constexpr, W_IN: tl.constexpr,
    C_OUT: tl.constexpr, D_OUT: tl.constexpr, H_OUT: tl.constexpr, W_OUT: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    S: tl.constexpr,
    eps,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    bias = tl.load(b_ptr + pid_c)
    m = tl.load(mult_ptr + pid_c)

    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    HW_OUT = H_OUT * W_OUT
    in_HW = H_IN * W_IN
    in_DHW = D_IN * in_HW

    # base pointers
    x_base = pid_n * C_IN * in_DHW
    out_base = pid_n * C_OUT * S + pid_c * S
    w_base = pid_c * C_IN * KD * KH * KW

    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S

        d_out = offs // HW_OUT
        rem = offs % HW_OUT
        h_out = rem // W_OUT
        w_out = rem % W_OUT

        acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

        for ci in tl.static_range(0, C_IN):
            for kd in tl.static_range(0, KD):
                for kh in tl.static_range(0, KH):
                    for kw in tl.static_range(0, KW):
                        d_in = d_out + kd
                        h_in = h_out + kh
                        w_in = w_out + kw
                        in_off = x_base + ci * in_DHW + d_in * in_HW + h_in * W_IN + w_in
                        wv = tl.load(w_ptr + w_base + ci * KD * KH * KW + kd * KH * KW + kh * KW + kw)
                        xv = tl.load(x_ptr + in_off, mask=mask, other=0.0)
                        acc += xv * wv

        acc += bias
        y = acc * m
        # store conv output (pre-mul, since we re-mul in pass 2 with multiplier)
        tl.store(conv_out_ptr + out_base + offs, acc, mask=mask)

        y_masked = tl.where(mask, y, 0.0)
        sum_val += tl.sum(y_masked, axis=0)
        sumsq_val += tl.sum(y_masked * y_masked, axis=0)

    mean = sum_val / S
    var = sumsq_val / S - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + pid_n * C_OUT + pid_c, mean)
    tl.store(invstd_ptr + pid_n * C_OUT + pid_c, inv_std)


@triton.jit
def fused_norm_clamp_max_kernel(
    x_ptr, mult_ptr, mean_ptr, invstd_ptr, out_ptr,
    N, C, S,
    clamp_min, clamp_max,
    BLOCK_S: tl.constexpr,
    C_CONST: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S

    neg_inf = float('-inf')
    acc = tl.full((BLOCK_S,), neg_inf, dtype=tl.float32)

    n_base = pid_n * C * S
    nc_base = pid_n * C

    for c in tl.static_range(0, C_CONST):
        m = tl.load(mult_ptr + c)
        mean = tl.load(mean_ptr + nc_base + c)
        invstd = tl.load(invstd_ptr + nc_base + c)
        x = tl.load(x_ptr + n_base + c * S + s_offs, mask=mask, other=0.0)
        y = x * m
        y = (y - mean) * invstd
        y = tl.minimum(tl.maximum(y, clamp_min), clamp_max)
        y = y * m
        acc = tl.maximum(acc, y)

    tl.store(out_ptr + pid_n * S + s_offs, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, C_IN, D_IN, H_IN, W_IN = x.shape
        KD = KH = KW = self.kernel_size
        C_OUT = self.out_channels
        D_OUT = D_IN - KD + 1
        H_OUT = H_IN - KH + 1
        W_OUT = W_IN - KW + 1
        S = D_OUT * H_OUT * W_OUT

        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()
        mult_flat = self.multiplier.contiguous().view(-1)

        conv_out = torch.empty((N, C_OUT, S), device=x.device, dtype=torch.float32)
        mean = torch.empty((N, C_OUT), device=x.device, dtype=torch.float32)
        invstd = torch.empty((N, C_OUT), device=x.device, dtype=torch.float32)

        BLOCK_S1 = 256
        conv3d_mean_invstd_kernel[(N, C_OUT)](
            x, weight, bias, mult_flat,
            conv_out, mean, invstd,
            N, C_IN, D_IN, H_IN, W_IN,
            C_OUT, D_OUT, H_OUT, W_OUT,
            KD, KH, KW,
            S,
            1e-5,
            BLOCK_S=BLOCK_S1,
            num_warps=4,
            num_stages=2,
        )

        out = torch.empty((N, D_OUT, H_OUT, W_OUT), device=x.device, dtype=torch.float32)
        out_flat = out.view(N, S)

        BLOCK_S2 = 512
        grid2 = (N, (S + BLOCK_S2 - 1) // BLOCK_S2)
        fused_norm_clamp_max_kernel[grid2](
            conv_out, mult_flat, mean, invstd, out_flat,
            N, C_OUT, S,
            self.clamp_min, self.clamp_max,
            BLOCK_S=BLOCK_S2,
            C_CONST=C_OUT,
            num_warps=4,
            num_stages=2,
        )

        return out