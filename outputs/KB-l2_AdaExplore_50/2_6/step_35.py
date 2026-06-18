import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_softmax_pool_kernel(
    x_ptr,         # input: (N, IC, ID, IH, IW)
    w_ptr,         # weight: (OC, IC, KD, KH, KW)
    b_ptr,         # bias: (OC,)
    out_ptr,       # output: (N, OC, Dp, Hp, Wp)
    N, IC, ID, IH, IW,
    OD, OH, OW,
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

    d_base = dp * POOL
    h_base = hp * POOL
    w_base = wp * POOL

    oc_offs = tl.arange(0, OC)
    bias = tl.load(b_ptr + oc_offs)

    ic_offs = tl.arange(0, IC_C)
    ic_mask = ic_offs < IC

    # Pre-load all weights into registers: shape [OC, IC_C * KD * KH * KW]
    # Total elements per program: OC * IC_C * KD * KH * KW = 16 * 4 * 27 = 1728
    # That's fine for registers.

    NEG_INF = float('-inf')
    max_acc = tl.full([OC], NEG_INF, dtype=tl.float32)

    x_stride_n = IC * ID * IH * IW
    x_stride_ic = ID * IH * IW
    x_stride_id = IH * IW
    x_stride_ih = IW

    # POOL = 4, so 4^3 = 64 spatial points per program
    for di in tl.static_range(0, POOL):
        for hi in tl.static_range(0, POOL):
            for wi in tl.static_range(0, POOL):
                od = d_base + di
                oh = h_base + hi
                ow = w_base + wi

                acc = bias

                for kd in tl.static_range(0, KD):
                    for kh in tl.static_range(0, KH):
                        for kw in tl.static_range(0, KW):
                            id_ = od + kd
                            ih_ = oh + kh
                            iw_ = ow + kw

                            x_base = n * x_stride_n + id_ * x_stride_id + ih_ * x_stride_ih + iw_
                            x_ptrs = x_ptr + x_base + ic_offs * x_stride_ic
                            x_vals = tl.load(x_ptrs, mask=ic_mask, other=0.0)  # [IC_C]

                            w_base_off = (kd * KH + kh) * KW + kw
                            w_ptrs = (
                                w_ptr
                                + oc_offs[:, None] * (IC * KD * KH * KW)
                                + ic_offs[None, :] * (KD * KH * KW)
                                + w_base_off
                            )
                            w_vals = tl.load(w_ptrs, mask=ic_mask[None, :], other=0.0)

                            acc += tl.sum(w_vals * x_vals[None, :], axis=1)

                # softmax over OC
                row_max = tl.max(acc, axis=0)
                exps = tl.exp(acc - row_max)
                row_sum = tl.sum(exps, axis=0)
                soft = exps / row_sum

                max_acc = tl.maximum(max_acc, soft)

    out_base = ((n * OC) * Dp + dp) * Hp * Wp + hp * Wp + wp
    out_stride_c = Dp * Hp * Wp
    out_ptrs = out_ptr + out_base + oc_offs * out_stride_c
    tl.store(out_ptrs, max_acc)


def fused_conv_softmax_pool(x, weight, bias, pool_combined):
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1
    POOL = pool_combined
    Dp = OD // POOL
    Hp = OH // POOL
    Wp = OW // POOL

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    out = torch.empty((N, OC, Dp, Hp, Wp), device=x.device, dtype=x.dtype)

    IC_C = 1
    while IC_C < IC:
        IC_C *= 2
    IC_C = max(IC_C, 4)

    grid = (N * Dp * Hp * Wp,)
    fused_conv_softmax_pool_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OD, OH, OW,
        Dp, Hp, Wp,
        POOL=POOL,
        OC=OC,
        KD=KD,
        KH=KH,
        KW=KW,
        IC_C=IC_C,
        num_warps=4,
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


def fused_softmax_pool(x, pool_combined):
    N, C, D, H, W = x.shape
    POOL = pool_combined
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
        x, out, N, C, D, H, W, Dp, Hp, Wp,
        POOL=POOL, BLOCK_C=BLOCK_C, num_warps=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        POOL = self.pool_kernel_size * self.pool_kernel_size
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1

        if (OD % POOL == 0 and OH % POOL == 0 and OW % POOL == 0
                and x.is_cuda and x.dtype == torch.float32):
            try:
                return fused_conv_softmax_pool(
                    x.contiguous(), self.conv.weight, self.conv.bias, POOL
                )
            except Exception:
                pass

        x = self.conv(x)
        N2, C2, D2, H2, W2 = x.shape
        if D2 % POOL == 0 and H2 % POOL == 0 and W2 % POOL == 0:
            return fused_softmax_pool(x, POOL)
        x = torch.softmax(x, dim=1)
        x = F.max_pool3d(x, self.pool_kernel_size)
        x = F.max_pool3d(x, self.pool_kernel_size)
        return x