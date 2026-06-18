import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_softmax_pool_kernel(
    x_ptr,        # input: (N, IC, D_in, H_in, W_in)
    w_ptr,        # weight: (OC, IC, KD, KH, KW)
    b_ptr,        # bias: (OC,)
    out_ptr,      # output: (N, OC, Dp, Hp, Wp)
    N, IC, D_in, H_in, W_in,
    OC, D_out, H_out, W_out,
    Dp, Hp, Wp,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid = tl.program_id(0)
    wp = pid % Wp
    t = pid // Wp
    hp = t % Hp
    t = t // Hp
    dp = t % Dp
    n = t // Dp

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Load bias for all OC
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    NEG_INF = float('-inf')
    max_acc = tl.full([BLOCK_OC], NEG_INF, dtype=tl.float32)

    d_base = dp * POOL
    h_base = hp * POOL
    w_base = wp * POOL

    # weight stride per OC
    w_per_oc = IC * KD * KH * KW
    # x stride per N
    x_per_n = IC * D_in * H_in * W_in
    x_per_ic = D_in * H_in * W_in

    # Iterate over POOL^3 spatial outputs (post-conv)
    for di in tl.static_range(0, POOL):
        for hi in tl.static_range(0, POOL):
            for wi in tl.static_range(0, POOL):
                d_out = d_base + di
                h_out = h_base + hi
                w_out = w_base + wi

                # Compute conv at (n, :, d_out, h_out, w_out) for all OC
                acc = tl.zeros([BLOCK_OC], dtype=tl.float32)

                for ic in tl.static_range(0, 3):  # IC=3
                    for kd in tl.static_range(0, KD):
                        for kh in tl.static_range(0, KH):
                            for kw in tl.static_range(0, KW):
                                d_in = d_out + kd
                                h_in = h_out + kh
                                w_in = w_out + kw
                                x_idx = (n * x_per_n
                                         + ic * x_per_ic
                                         + d_in * H_in * W_in
                                         + h_in * W_in
                                         + w_in)
                                xv = tl.load(x_ptr + x_idx)
                                w_idx_base = (oc_offs * w_per_oc
                                              + ic * KD * KH * KW
                                              + kd * KH * KW
                                              + kh * KW
                                              + kw)
                                wv = tl.load(w_ptr + w_idx_base, mask=oc_mask, other=0.0)
                                acc += xv * wv

                acc = acc + bias
                # softmax over OC
                acc_masked = tl.where(oc_mask, acc, NEG_INF)
                row_max = tl.max(acc_masked, axis=0)
                exps = tl.exp(acc_masked - row_max)
                exps = tl.where(oc_mask, exps, 0.0)
                row_sum = tl.sum(exps, axis=0)
                soft = exps / row_sum

                max_acc = tl.maximum(max_acc, soft)

    out_base = ((n * OC + 0) * Dp + dp) * Hp * Wp + hp * Wp + wp
    out_stride_c = Dp * Hp * Wp
    out_ptrs = out_ptr + out_base + oc_offs * out_stride_c
    tl.store(out_ptrs, max_acc, mask=oc_mask)


def fused_conv_softmax_pool(x, weight, bias, pool_kernel_size):
    N, IC, D_in, H_in, W_in = x.shape
    OC, _, KD, KH, KW = weight.shape
    D_out = D_in - KD + 1
    H_out = H_in - KH + 1
    W_out = W_in - KW + 1
    POOL = pool_kernel_size * pool_kernel_size
    Dp = D_out // POOL
    Hp = H_out // POOL
    Wp = W_out // POOL

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    out = torch.empty((N, OC, Dp, Hp, Wp), device=x.device, dtype=x.dtype)

    BLOCK_OC = 1
    while BLOCK_OC < OC:
        BLOCK_OC *= 2
    BLOCK_OC = max(BLOCK_OC, 16)

    grid = (N * Dp * Hp * Wp,)
    fused_conv_softmax_pool_kernel[grid](
        x, weight, bias, out,
        N, IC, D_in, H_in, W_in,
        OC, D_out, H_out, W_out,
        Dp, Hp, Wp,
        KD=KD, KH=KH, KW=KW,
        POOL=POOL,
        BLOCK_OC=BLOCK_OC,
        num_warps=2,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        POOL = self.pool_kernel_size * self.pool_kernel_size
        N, IC, D, H, W = x.shape
        D_out = D - self.kernel_size + 1
        H_out = H - self.kernel_size + 1
        W_out = W - self.kernel_size + 1
        if (D_out % POOL == 0 and H_out % POOL == 0 and W_out % POOL == 0
                and IC == 3):
            return fused_conv_softmax_pool(
                x, self.conv.weight, self.conv.bias, self.pool_kernel_size)
        else:
            x = self.conv(x)
            x = torch.softmax(x, dim=1)
            x = F.max_pool3d(x, self.pool_kernel_size)
            x = F.max_pool3d(x, self.pool_kernel_size)
            return x