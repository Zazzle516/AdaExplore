import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, conv_bias_ptr, extra_bias_ptr, out_ptr,
    N, IC, D_in, H_in, W_in,
    OC, D_out, H_out, W_out,
    KD, KH, KW,
    HW_out, DHW_out,
    HW_in, DHW_in,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_TOTAL: tl.constexpr,
):
    pid_m = tl.program_id(0)  # spatial tile
    pid_n = tl.program_id(1)  # OC tile
    pid_b = tl.program_id(2)  # batch index

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # output spatial linear index
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # OC index

    # Decompose offs_m into (d_out, h_out, w_out)
    d_out = offs_m // HW_out
    rem = offs_m % HW_out
    h_out = rem // W_out
    w_out = rem % W_out

    m_mask = offs_m < DHW_out
    n_mask = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    # Loop over K = IC * KD * KH * KW
    for k_start in range(0, K_TOTAL, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]
        k_mask = k_idx < K_TOTAL

        # decompose k_idx into (ic, kd, kh, kw)
        # k = ((ic * KD + kd) * KH + kh) * KW + kw
        kw = k_idx % KW
        tmp1 = k_idx // KW
        kh = tmp1 % KH
        tmp2 = tmp1 // KH
        kd = tmp2 % KD
        ic = tmp2 // KD

        # Input gather: x[b, ic, d_out+kd, h_out+kh, w_out+kw]
        d_in = d_out[:, None] + kd[None, :]
        h_in = h_out[:, None] + kh[None, :]
        w_in = w_out[:, None] + kw[None, :]

        x_offsets = (pid_b * IC * DHW_in
                     + ic[None, :] * DHW_in
                     + d_in * HW_in
                     + h_in * W_in
                     + w_in)
        x_mask = m_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

        # Weight gather: w[oc, ic, kd, kh, kw]
        # weight stride: (IC*KD*KH*KW, KD*KH*KW, KH*KW, KW, 1)
        w_offsets = offs_n[:, None] * K_TOTAL + k_idx[None, :]
        w_mask = n_mask[:, None] & k_mask[None, :]
        w_vals = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)

        # Need [BLOCK_M, BLOCK_K] x [BLOCK_K, BLOCK_N]
        acc += tl.dot(x_vals, tl.trans(w_vals))

    # Add conv bias
    cb = tl.load(conv_bias_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + cb[None, :]

    # Activations: ReLU -> LeakyReLU(0.01) -> GELU -> Sigmoid -> + extra_bias
    # After ReLU, x >= 0, LeakyReLU is identity
    acc = tl.maximum(acc, 0.0)
    # GELU tanh approx
    k0 = 0.7978845608028654
    k1 = 0.044715
    inner = k0 * (acc + k1 * acc * acc * acc)
    e2 = tl.exp(2.0 * inner)
    tanh_val = (e2 - 1.0) / (e2 + 1.0)
    gelu = 0.5 * acc * (1.0 + tanh_val)
    sig = 1.0 / (1.0 + tl.exp(-gelu))

    eb = tl.load(extra_bias_ptr + offs_n, mask=n_mask, other=0.0)
    out_vals = sig + eb[None, :]

    # Store: out[b, oc, d_out, h_out, w_out]
    out_offsets = (pid_b * OC * DHW_out
                   + offs_n[None, :] * DHW_out
                   + offs_m[:, None])
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_offsets, out_vals, mask=out_mask)


def conv3d_fused(x, weight, conv_bias, extra_bias):
    x = x.contiguous()
    weight = weight.contiguous()
    N, IC, D_in, H_in, W_in = x.shape
    OC, _, KD, KH, KW = weight.shape
    D_out = D_in - KD + 1
    H_out = H_in - KH + 1
    W_out = W_in - KW + 1

    out = torch.empty((N, OC, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    DHW_out = D_out * H_out * W_out
    HW_out = H_out * W_out
    DHW_in = D_in * H_in * W_in
    HW_in = H_in * W_in
    K_TOTAL = IC * KD * KH * KW

    BLOCK_M = 64
    BLOCK_N = 32
    BLOCK_K = 32

    extra_bias_flat = extra_bias.contiguous().view(-1)

    grid = (
        (DHW_out + BLOCK_M - 1) // BLOCK_M,
        (OC + BLOCK_N - 1) // BLOCK_N,
        N,
    )

    conv3d_fused_kernel[grid](
        x, weight, conv_bias, extra_bias_flat, out,
        N, IC, D_in, H_in, W_in,
        OC, D_out, H_out, W_out,
        KD, KH, KW,
        HW_out, DHW_out,
        HW_in, DHW_in,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        K_TOTAL=K_TOTAL,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        return conv3d_fused(x, self.conv.weight, self.conv.bias, self.bias)