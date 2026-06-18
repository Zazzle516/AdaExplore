import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_softmax_pool_kernel(
    x_ptr,          # input: (N, IC, D_in, H_in, W_in)
    w_ptr,          # weight: (OC, IC, KD, KH, KW)
    b_ptr,          # bias: (OC,)
    out_ptr,        # output: (N, OC, Dp, Hp, Wp)
    N, IC,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    Dp, Hp, Wp,
    OC: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid = tl.program_id(0)
    wp = pid % Wp
    pid2 = pid // Wp
    hp = pid2 % Hp
    pid3 = pid2 // Hp
    dp = pid3 % Dp
    n = pid3 // Dp

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Load bias once
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    NEG_INF = float('-inf')
    max_acc = tl.full([BLOCK_OC], NEG_INF, dtype=tl.float32)

    d_base = dp * POOL
    h_base = hp * POOL
    w_base = wp * POOL

    x_stride_n = IC * D_in * H_in * W_in
    x_stride_c = D_in * H_in * W_in
    x_stride_d = H_in * W_in
    x_stride_h = W_in

    w_stride_oc = IC * KD * KH * KW
    w_stride_ic = KD * KH * KW
    w_stride_kd = KH * KW
    w_stride_kh = KW

    # Iterate over POOL^3 spatial window of conv output
    for di in tl.static_range(0, POOL):
        for hi in tl.static_range(0, POOL):
            for wi in tl.static_range(0, POOL):
                od = d_base + di
                oh = h_base + hi
                ow = w_base + wi

                # Compute conv output for all OC at (n, :, od, oh, ow)
                acc = tl.zeros([BLOCK_OC], dtype=tl.float32)

                for ic in tl.static_range(0, 3):  # IC = 3
                    for kd in tl.static_range(0, KD):
                        for kh in tl.static_range(0, KH):
                            for kw in tl.static_range(0, KW):
                                id_ = od + kd
                                ih = oh + kh
                                iw = ow + kw
                                x_off = (n * x_stride_n
                                         + ic * x_stride_c
                                         + id_ * x_stride_d
                                         + ih * x_stride_h
                                         + iw)
                                xv = tl.load(x_ptr + x_off)
                                w_off = (oc_offs * w_stride_oc
                                         + ic * w_stride_ic
                                         + kd * w_stride_kd
                                         + kh * w_stride_kh
                                         + kw)
                                wv = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                                acc = acc + xv * wv

                acc = acc + bias
                acc = tl.where(oc_mask, acc, NEG_INF)

                # Softmax over OC
                row_max = tl.max(acc, axis=0)
                exps = tl.exp(acc - row_max)
                exps = tl.where(oc_mask, exps, 0.0)
                row_sum = tl.sum(exps, axis=0)
                soft = exps / row_sum

                max_acc = tl.maximum(max_acc, soft)

    out_stride_c = Dp * Hp * Wp
    out_base = n * OC * out_stride_c + dp * Hp * Wp + hp * Wp + wp
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
        N, IC,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        Dp, Hp, Wp,
        OC=OC, KD=KD, KH=KH, KW=KW,
        POOL=POOL,
        BLOCK_OC=BLOCK_OC,
        num_warps=4,
    )
    return out


@triton.jit
def fused_softmax_pool_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    Dp, Hp, Wp,
    POOL: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    wp = pid % Wp
    pid2 = pid // Wp
    hp = pid2 % Hp
    pid3 = pid2 // Hp
    dp = pid3 % Dp
    n = pid3 // Dp

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    NEG_INF = float('-inf')
    max_acc = tl.full([BLOCK_C], NEG_INF, dtype=tl.float32)

    d_base = dp * POOL
    h_base = hp * POOL
    w_base = wp * POOL

    for di in tl.static_range(0, POOL):
        for hi in tl.static_range(0, POOL):
            for wi in tl.static_range(0, POOL):
                d_idx = d_base + di
                h_idx = h_base + hi
                w_idx = w_base + wi

                base = ((n * C + 0) * D + d_idx) * H * W + h_idx * W + w_idx
                stride_c = D * H * W
                ptrs = x_ptr + base + c_offs * stride_c

                vals = tl.load(ptrs, mask=c_mask, other=NEG_INF)
                row_max = tl.max(vals, axis=0)
                exps = tl.exp(vals - row_max)
                exps = tl.where(c_mask, exps, 0.0)
                row_sum = tl.sum(exps, axis=0)
                soft = exps / row_sum

                max_acc = tl.maximum(max_acc, soft)

    out_base = ((n * C + 0) * Dp + dp) * Hp * Wp + hp * Wp + wp
    out_stride_c = Dp * Hp * Wp
    out_ptrs = out_ptr + out_base + c_offs * out_stride_c
    tl.store(out_ptrs, max_acc, mask=c_mask)


def fused_softmax_pool(x: torch.Tensor, pool_kernel_size: int):
    N, C, D, H, W = x.shape
    POOL = pool_kernel_size * pool_kernel_size
    Dp = D // POOL
    Hp = H // POOL
    Wp = W // POOL

    x = x.contiguous()
    out = torch.empty((N, C, Dp, Hp, Wp), device=x.device, dtype=x.dtype)

    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2
    BLOCK_C = max(BLOCK_C, 16)

    grid = (N * Dp * Hp * Wp,)
    fused_softmax_pool_kernel[grid](
        x, out,
        N, C, D, H, W,
        Dp, Hp, Wp,
        POOL=POOL,
        BLOCK_C=BLOCK_C,
        num_warps=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        POOL = self.pool_kernel_size * self.pool_kernel_size
        N, IC, D_in, H_in, W_in = x.shape
        KD = KH = KW = self.kernel_size
        D_out = D_in - KD + 1
        H_out = H_in - KH + 1
        W_out = W_in - KW + 1

        # Try fully fused conv+softmax+pool when shapes match
        if (IC == 3 and D_out % POOL == 0 and H_out % POOL == 0 and W_out % POOL == 0
                and x.is_cuda):
            return fused_conv_softmax_pool(
                x, self.conv.weight, self.conv.bias, self.pool_kernel_size
            )

        # Otherwise fall back to conv + fused softmax/pool
        x = self.conv(x)
        N, C, D, H, W = x.shape
        if D % POOL == 0 and H % POOL == 0 and W % POOL == 0 and x.is_cuda:
            return fused_softmax_pool(x, self.pool_kernel_size)
        else:
            x = torch.softmax(x, dim=1)
            x = F.max_pool3d(x, self.pool_kernel_size)
            x = F.max_pool3d(x, self.pool_kernel_size)
            return x