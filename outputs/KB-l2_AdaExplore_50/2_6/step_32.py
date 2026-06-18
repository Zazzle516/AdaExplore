import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_softmax_pool_kernel(
    x_ptr,           # input: (N, IC, D_in, H_in, W_in)
    w_ptr,           # weight reshaped: (IC*KD*KH*KW, OC)
    b_ptr,           # bias: (OC,)
    out_ptr,         # output: (N, OC, Dp, Hp, Wp)
    N, IC,
    D_in, H_in, W_in,
    Dp, Hp, Wp,
    POOL: tl.constexpr,
    OC: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    IC_C: tl.constexpr,
):
    pid = tl.program_id(0)
    wp = pid % Wp
    pid2 = pid // Wp
    hp = pid2 % Hp
    pid3 = pid2 // Hp
    dp = pid3 % Dp
    n = pid3 // Dp

    c_offs = tl.arange(0, OC)

    NEG_INF = float('-inf')
    max_acc = tl.full([OC], NEG_INF, dtype=tl.float32)

    bias = tl.load(b_ptr + c_offs)

    d_base = dp * POOL
    h_base = hp * POOL
    w_base = wp * POOL

    in_chw = D_in * H_in * W_in
    in_hw = H_in * W_in
    KVOL = KD * KH * KW
    # weight is now (KVOL*IC, OC) row-major: row index = (ic*KVOL + k), col = oc
    # but we stored as (IC*KVOL, OC) with row = ic*KVOL + (kd*KH*KW + kh*KW + kw)

    n_off = n * (IC * in_chw)

    for di in tl.static_range(0, POOL):
        for hi in tl.static_range(0, POOL):
            for wi in tl.static_range(0, POOL):
                od = d_base + di
                oh = h_base + hi
                ow = w_base + wi

                acc = bias

                for ic in tl.static_range(0, IC_C):
                    ic_x_base = n_off + ic * in_chw
                    w_ic_base = ic * KVOL
                    for kd in tl.static_range(0, KD):
                        id_ = od + kd
                        for kh in tl.static_range(0, KH):
                            ih_ = oh + kh
                            x_row_base = ic_x_base + id_ * in_hw + ih_ * W_in + ow
                            w_row_base = (w_ic_base + kd * (KH * KW) + kh * KW) * OC
                            for kw in tl.static_range(0, KW):
                                xv = tl.load(x_ptr + x_row_base + kw)
                                w_off = w_row_base + kw * OC + c_offs
                                wv = tl.load(w_ptr + w_off)
                                acc = acc + xv * wv

                row_max = tl.max(acc, axis=0)
                exps = tl.exp(acc - row_max)
                row_sum = tl.sum(exps, axis=0)
                soft = exps / row_sum

                max_acc = tl.maximum(max_acc, soft)

    out_base = n * (OC * Dp * Hp * Wp) + dp * (Hp * Wp) + hp * Wp + wp
    out_stride_c = Dp * Hp * Wp
    out_ptrs = out_ptr + out_base + c_offs * out_stride_c
    tl.store(out_ptrs, max_acc)


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
    # weight shape (OC, IC, KD, KH, KW) -> (IC, KD, KH, KW, OC) -> (IC*KD*KH*KW, OC)
    w_t = weight.permute(1, 2, 3, 4, 0).contiguous().view(IC * KD * KH * KW, OC)
    bias = bias.contiguous()

    out = torch.empty((N, OC, Dp, Hp, Wp), device=x.device, dtype=torch.float32)

    grid = (N * Dp * Hp * Wp,)
    fused_conv_softmax_pool_kernel[grid](
        x, w_t, bias, out,
        N, IC,
        D_in, H_in, W_in,
        Dp, Hp, Wp,
        POOL=POOL,
        OC=OC,
        KD=KD, KH=KH, KW=KW,
        IC_C=IC,
        num_warps=2,
        num_stages=2,
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


def fused_softmax_pool(x, pool_kernel_size):
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
        weight = self.conv.weight
        bias = self.conv.bias

        N, IC, D_in, H_in, W_in = x.shape
        OC, _, KD, KH, KW = weight.shape
        D_out = D_in - KD + 1
        H_out = H_in - KH + 1
        W_out = W_in - KW + 1

        oc_pow2 = (OC & (OC - 1)) == 0
        if (oc_pow2 and D_out % POOL == 0 and H_out % POOL == 0 and W_out % POOL == 0
                and x.is_cuda):
            x = x.contiguous()
            return fused_conv_softmax_pool(x, weight, bias, self.pool_kernel_size)
        else:
            x = self.conv(x)
            if D_out % POOL == 0 and H_out % POOL == 0 and W_out % POOL == 0:
                return fused_softmax_pool(x, self.pool_kernel_size)
            x = torch.softmax(x, dim=1)
            x = F.max_pool3d(x, self.pool_kernel_size)
            x = F.max_pool3d(x, self.pool_kernel_size)
            return x