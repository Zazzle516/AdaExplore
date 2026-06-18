import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, D_in, H_in, W_in,
    C_out, D_out, H_out, W_out,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    C_IN_CONST: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_s = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    oc_mask = oc_offs < C_out

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < (D_out * H_out * W_out)

    od = s_offs // (H_out * W_out)
    rem = s_offs % (H_out * W_out)
    oh = rem // W_out
    ow = rem % W_out

    # Accumulator [BLOCK_OC, BLOCK_S]
    acc = tl.zeros([BLOCK_OC, BLOCK_S], dtype=tl.float32)

    x_batch_base = pid_n * C_in * D_in * H_in * W_in

    for kd in tl.static_range(0, KD):
        id_ = od + kd  # [BLOCK_S]
        for kh in tl.static_range(0, KH):
            ih = oh + kh
            for kw in tl.static_range(0, KW):
                iw = ow + kw
                # weight: [C_out, C_in, KD, KH, KW]
                # offset for (oc, ic, kd, kh, kw): oc*C_in*KD*KH*KW + ic*KD*KH*KW + kd*KH*KW + kh*KW + kw
                w_kernel_off = kd * KH * KW + kh * KW + kw  # scalar

                # Load weight slice [BLOCK_OC, C_IN_CONST]
                ic_offs = tl.arange(0, C_IN_CONST)
                ic_mask = ic_offs < C_in
                w_ptrs = w_ptr + oc_offs[:, None] * (C_in * KD * KH * KW) + ic_offs[None, :] * (KD * KH * KW) + w_kernel_off
                w_mask = oc_mask[:, None] & ic_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_OC, C_IN_CONST]

                # Load input slice [C_IN_CONST, BLOCK_S]
                x_spatial_off = id_[None, :] * (H_in * W_in) + ih[None, :] * W_in + iw[None, :]  # [1, BLOCK_S]
                x_ptrs = x_ptr + x_batch_base + ic_offs[:, None] * (D_in * H_in * W_in) + x_spatial_off
                x_load_mask = ic_mask[:, None] & s_mask[None, :]
                x_vals = tl.load(x_ptrs, mask=x_load_mask, other=0.0)  # [C_IN_CONST, BLOCK_S]

                acc += tl.dot(w_vals, x_vals, allow_tf32=False)

    # Add bias
    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc += b_vals[:, None]

    # Store output [N, C_out, S_out]
    S_out = D_out * H_out * W_out
    out_ptrs = out_ptr + pid_n * C_out * S_out + oc_offs[:, None] * S_out + s_offs[None, :]
    out_store_mask = oc_mask[:, None] & s_mask[None, :]
    tl.store(out_ptrs, acc, mask=out_store_mask)


@triton.jit
def stats_kernel(
    x_ptr, mult_ptr, mean_ptr, invstd_ptr,
    N, C, S,
    eps,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    m = tl.load(mult_ptr + c)
    base = n * C * S + c * S

    sum1 = tl.zeros([BLOCK_S], dtype=tl.float32)
    sum2 = tl.zeros([BLOCK_S], dtype=tl.float32)
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = x * m
        y = tl.where(mask, y, 0.0)
        sum1 += y
        sum2 += y * y

    s1 = tl.sum(sum1, axis=0)
    s2 = tl.sum(sum2, axis=0)
    mean = s1 / S
    var = s2 / S - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + n * C + c, mean)
    tl.store(invstd_ptr + n * C + c, invstd)


@triton.jit
def apply_and_max_kernel(
    x_ptr, mult_ptr, mean_ptr, invstd_ptr, out_ptr,
    N, C, S,
    clamp_min, clamp_max,
    BLOCK_S: tl.constexpr,
    C_CONST: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    NEG_INF = -float('inf')
    max_val = tl.full([BLOCK_S], NEG_INF, dtype=tl.float32)

    for c_i in tl.static_range(0, C_CONST):
        if c_i < C:
            m = tl.load(mult_ptr + c_i)
            mu = tl.load(mean_ptr + pid_n * C + c_i)
            iv = tl.load(invstd_ptr + pid_n * C + c_i)
            x = tl.load(x_ptr + pid_n * C * S + c_i * S + s_offs, mask=s_mask, other=0.0)
            y = x * m
            y = (y - mu) * iv
            y = tl.minimum(tl.maximum(y, clamp_min), clamp_max)
            y = y * m
            max_val = tl.maximum(max_val, y)

    tl.store(out_ptr + pid_n * S + s_offs, max_val, mask=s_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super().__init__()
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
        N, C_in, D_in, H_in, W_in = x.shape
        KD = KH = KW = self.kernel_size
        C_out = self.out_channels
        D_out = D_in - KD + 1
        H_out = H_in - KH + 1
        W_out = W_in - KW + 1
        S_out = D_out * H_out * W_out

        # Allocate conv output
        conv_out = torch.empty((N, C_out, S_out), device=x.device, dtype=torch.float32)

        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()

        BLOCK_OC = 16
        BLOCK_S = 128
        # C_IN_CONST must be power of 2 >= C_in
        C_IN_CONST = 1
        while C_IN_CONST < C_in:
            C_IN_CONST *= 2
        if C_IN_CONST < 16:
            C_IN_CONST = 16  # tl.dot needs minimum sizes

        grid = (N, triton.cdiv(C_out, BLOCK_OC), triton.cdiv(S_out, BLOCK_S))
        conv3d_kernel[grid](
            x, weight, bias, conv_out,
            N, C_in, D_in, H_in, W_in,
            C_out, D_out, H_out, W_out,
            KD, KH, KW,
            C_IN_CONST,
            BLOCK_OC, BLOCK_S,
            num_warps=4, num_stages=2,
        )

        mult_flat = self.multiplier.contiguous().view(-1)

        mean = torch.empty((N, C_out), device=x.device, dtype=torch.float32)
        invstd = torch.empty((N, C_out), device=x.device, dtype=torch.float32)

        STATS_BLOCK = 1024
        stats_kernel[(N * C_out,)](
            conv_out, mult_flat, mean, invstd,
            N, C_out, S_out, 1e-5,
            BLOCK_S=STATS_BLOCK,
            num_warps=4,
        )

        out = torch.empty((N, S_out), device=x.device, dtype=torch.float32)
        C_CONST = 1
        while C_CONST < C_out:
            C_CONST *= 2

        APPLY_BLOCK = 1024
        grid2 = (N, triton.cdiv(S_out, APPLY_BLOCK))
        apply_and_max_kernel[grid2](
            conv_out, mult_flat, mean, invstd, out,
            N, C_out, S_out,
            self.clamp_min, self.clamp_max,
            BLOCK_S=APPLY_BLOCK,
            C_CONST=C_CONST,
            num_warps=4,
        )

        return out.view(N, D_out, H_out, W_out)