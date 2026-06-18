import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=1, num_stages=3),
        triton.Config({}, num_warps=1, num_stages=4),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=2, num_stages=4),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'KD', 'KH', 'KW', 'IC_C', 'POOL'],
)
@triton.jit
def fused_conv_softmax_pool_kernel(
    x_ptr,           # input: (N, IC, D_in, H_in, W_in)
    w_ptr,           # weight: (OC, IC, KD, KH, KW)
    b_ptr,           # bias: (OC,)
    out_ptr,         # output: (N, OC, Dp, Hp, Wp)
    N, IC,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    Dp, Hp, Wp,
    POOL: tl.constexpr,        # combined pool window
    OC: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    IC_C: tl.constexpr,
    K_SIZE: tl.constexpr,      # next_pow2(IC*KD*KH*KW)
):
    pid = tl.program_id(0)
    wp = pid % Wp
    pid2 = pid // Wp
    hp = pid2 % Hp
    pid3 = pid2 // Hp
    dp = pid3 % Dp
    n = pid3 // Dp

    c_offs = tl.arange(0, OC)
    k_offs = tl.arange(0, K_SIZE)
    K_REAL = IC_C * KD * KH * KW
    k_mask = k_offs < K_REAL

    NEG_INF = float('-inf')
    max_acc = tl.full([OC], NEG_INF, dtype=tl.float32)

    # Load bias
    bias = tl.load(b_ptr + c_offs)

    # Load full weight tile [OC, K_SIZE] once
    w_ptrs = w_ptr + c_offs[:, None] * K_REAL + k_offs[None, :]
    w_tile = tl.load(w_ptrs, mask=k_mask[None, :], other=0.0)  # [OC, K_SIZE]

    # Decode k -> (ic, kd, kh, kw)
    KHW = KH * KW
    KDHW = KD * KHW
    ic_idx = k_offs // KDHW
    rem = k_offs % KDHW
    kd_idx = rem // KHW
    rem2 = rem % KHW
    kh_idx = rem2 // KW
    kw_idx = rem2 % KW

    in_chw = D_in * H_in * W_in
    in_hw = H_in * W_in
    n_off = n * IC * in_chw

    d_base = dp * POOL
    h_base = hp * POOL
    w_base = wp * POOL

    # Precompute input offset components that don't depend on (di, hi, wi)
    in_k_base = n_off + ic_idx * in_chw + kd_idx * in_hw + kh_idx * W_in + kw_idx

    # Loop over POOL^3 spatial points
    for di in tl.static_range(0, POOL):
        for hi in tl.static_range(0, POOL):
            for wi in tl.static_range(0, POOL):
                od = d_base + di
                oh = h_base + hi
                ow = w_base + wi
                spatial_off = od * in_hw + oh * W_in + ow
                in_ptrs = x_ptr + in_k_base + spatial_off
                xv = tl.load(in_ptrs, mask=k_mask, other=0.0)  # [K_SIZE]
                # matvec: [OC, K] * [K] -> [OC]
                acc = tl.sum(w_tile * xv[None, :], axis=1) + bias

                # Softmax over OC
                row_max = tl.max(acc, axis=0)
                exps = tl.exp(acc - row_max)
                row_sum = tl.sum(exps, axis=0)
                soft = exps / row_sum

                max_acc = tl.maximum(max_acc, soft)

    out_base = n * (OC * Dp * Hp * Wp) + dp * (Hp * Wp) + hp * Wp + wp
    out_stride_c = Dp * Hp * Wp
    out_ptrs = out_ptr + out_base + c_offs * out_stride_c
    tl.store(out_ptrs, max_acc)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


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

    out = torch.empty((N, OC, Dp, Hp, Wp), device=x.device, dtype=torch.float32)

    K_REAL = IC * KD * KH * KW
    K_SIZE = _next_pow2(K_REAL)

    grid = (N * Dp * Hp * Wp,)
    fused_conv_softmax_pool_kernel[grid](
        x, weight, bias, out,
        N, IC,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        Dp, Hp, Wp,
        POOL=POOL,
        OC=OC,
        KD=KD, KH=KH, KW=KW,
        IC_C=IC,
        K_SIZE=K_SIZE,
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

        # Check that OC is power of 2 and dims are divisible
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