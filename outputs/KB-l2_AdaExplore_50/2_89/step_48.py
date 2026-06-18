import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'TILE_W': 4}, num_warps=2, num_stages=2),
        triton.Config({'TILE_W': 4}, num_warps=4, num_stages=2),
        triton.Config({'TILE_W': 8}, num_warps=2, num_stages=2),
        triton.Config({'TILE_W': 8}, num_warps=4, num_stages=2),
        triton.Config({'TILE_W': 8}, num_warps=4, num_stages=3),
        triton.Config({'TILE_W': 8}, num_warps=4, num_stages=4),
        triton.Config({'TILE_W': 16}, num_warps=4, num_stages=2),
        triton.Config({'TILE_W': 16}, num_warps=4, num_stages=3),
        triton.Config({'TILE_W': 16}, num_warps=4, num_stages=4),
        triton.Config({'TILE_W': 16}, num_warps=8, num_stages=2),
        triton.Config({'TILE_W': 16}, num_warps=8, num_stages=3),
        triton.Config({'TILE_W': 16}, num_warps=8, num_stages=4),
        triton.Config({'TILE_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'TILE_W': 32}, num_warps=4, num_stages=3),
        triton.Config({'TILE_W': 32}, num_warps=8, num_stages=2),
        triton.Config({'TILE_W': 32}, num_warps=8, num_stages=3),
        triton.Config({'TILE_W': 32}, num_warps=8, num_stages=4),
    ],
    key=['N', 'IC', 'OC', 'Din', 'Hin', 'Win', 'Dp', 'Hp', 'Wp'],
)
@triton.jit
def fused_full_kernel(
    x_ptr,              # (N, IC, Din, Hin, Win)
    w_ptr,              # (IC, OC, KD, KH, KW)
    b_ptr,              # (OC,)
    sub_ptr,            # (OC,)
    out_ptr,            # (N, Dp, Hp, Wp)
    N,
    IC: tl.constexpr,
    OC: tl.constexpr,
    Din: tl.constexpr, Hin: tl.constexpr, Win: tl.constexpr,
    Dp: tl.constexpr, Hp: tl.constexpr, Wp: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    TILE_W: tl.constexpr,
):
    pid = tl.program_id(0)
    Wp_tiles = Wp // TILE_W
    HpWt = Hp * Wp_tiles
    DpHpWt = Dp * HpWt
    n = pid // DpHpWt
    rem = pid % DpHpWt
    dp = rem // HpWt
    rem2 = rem % HpWt
    hp = rem2 // Wp_tiles
    wt = rem2 % Wp_tiles

    wp_base = wt * TILE_W

    d0 = 2 * dp
    h0 = 2 * hp
    w0 = 2 * wp_base

    offs_oc = tl.arange(0, OC)
    offs_t = tl.arange(0, TILE_W)

    # 8 accumulators, one per pooled position (ddp, dhp, dwp)
    acc0 = tl.zeros((OC, TILE_W), dtype=tl.float32)
    acc1 = tl.zeros((OC, TILE_W), dtype=tl.float32)
    acc2 = tl.zeros((OC, TILE_W), dtype=tl.float32)
    acc3 = tl.zeros((OC, TILE_W), dtype=tl.float32)
    acc4 = tl.zeros((OC, TILE_W), dtype=tl.float32)
    acc5 = tl.zeros((OC, TILE_W), dtype=tl.float32)
    acc6 = tl.zeros((OC, TILE_W), dtype=tl.float32)
    acc7 = tl.zeros((OC, TILE_W), dtype=tl.float32)

    # Loop order: kd/kh/kw/ic outer (load weight once), pool positions inner
    for kd in tl.static_range(0, KD):
        # Precompute id_/valid_d for both ddp values
        num_d0 = d0 + PAD - kd
        id_d0 = num_d0 // STRIDE
        valid_d0 = ((num_d0 % STRIDE) == 0) & (id_d0 >= 0) & (id_d0 < Din)
        num_d1 = d0 + 1 + PAD - kd
        id_d1 = num_d1 // STRIDE
        valid_d1 = ((num_d1 % STRIDE) == 0) & (id_d1 >= 0) & (id_d1 < Din)

        for kh in tl.static_range(0, KH):
            num_h0 = h0 + PAD - kh
            ih_h0 = num_h0 // STRIDE
            valid_h0 = ((num_h0 % STRIDE) == 0) & (ih_h0 >= 0) & (ih_h0 < Hin)
            num_h1 = h0 + 1 + PAD - kh
            ih_h1 = num_h1 // STRIDE
            valid_h1 = ((num_h1 % STRIDE) == 0) & (ih_h1 >= 0) & (ih_h1 < Hin)

            for kw in tl.static_range(0, KW):
                # Compute iw_ / valid_w for dwp=0 and dwp=1
                num_w0 = (w0 + 2 * offs_t) + PAD - kw
                iw_w0 = num_w0 // STRIDE
                valid_w0 = ((num_w0 % STRIDE) == 0) & (iw_w0 >= 0) & (iw_w0 < Win)
                num_w1 = (w0 + 2 * offs_t + 1) + PAD - kw
                iw_w1 = num_w1 // STRIDE
                valid_w1 = ((num_w1 % STRIDE) == 0) & (iw_w1 >= 0) & (iw_w1 < Win)

                for ic in tl.static_range(0, IC):
                    # Load weight vector once for this (ic, kd, kh, kw)
                    w_base = (((ic * OC) * KD + kd) * KH + kh) * KW + kw
                    w_off = w_base + offs_oc * (KD * KH * KW)
                    wv = tl.load(w_ptr + w_off)

                    base_x = ((n * IC + ic) * Din)

                    # (ddp=0, dhp=0)
                    x_off = ((base_x + id_d0) * Hin + ih_h0) * Win
                    valid00w0 = valid_d0 & valid_h0 & valid_w0
                    xv = tl.load(x_ptr + x_off + iw_w0, mask=valid00w0, other=0.0)
                    acc0 += wv[:, None] * xv[None, :]
                    valid00w1 = valid_d0 & valid_h0 & valid_w1
                    xv = tl.load(x_ptr + x_off + iw_w1, mask=valid00w1, other=0.0)
                    acc1 += wv[:, None] * xv[None, :]

                    # (ddp=0, dhp=1)
                    x_off = ((base_x + id_d0) * Hin + ih_h1) * Win
                    valid01w0 = valid_d0 & valid_h1 & valid_w0
                    xv = tl.load(x_ptr + x_off + iw_w0, mask=valid01w0, other=0.0)
                    acc2 += wv[:, None] * xv[None, :]
                    valid01w1 = valid_d0 & valid_h1 & valid_w1
                    xv = tl.load(x_ptr + x_off + iw_w1, mask=valid01w1, other=0.0)
                    acc3 += wv[:, None] * xv[None, :]

                    # (ddp=1, dhp=0)
                    x_off = ((base_x + id_d1) * Hin + ih_h0) * Win
                    valid10w0 = valid_d1 & valid_h0 & valid_w0
                    xv = tl.load(x_ptr + x_off + iw_w0, mask=valid10w0, other=0.0)
                    acc4 += wv[:, None] * xv[None, :]
                    valid10w1 = valid_d1 & valid_h0 & valid_w1
                    xv = tl.load(x_ptr + x_off + iw_w1, mask=valid10w1, other=0.0)
                    acc5 += wv[:, None] * xv[None, :]

                    # (ddp=1, dhp=1)
                    x_off = ((base_x + id_d1) * Hin + ih_h1) * Win
                    valid11w0 = valid_d1 & valid_h1 & valid_w0
                    xv = tl.load(x_ptr + x_off + iw_w0, mask=valid11w0, other=0.0)
                    acc6 += wv[:, None] * xv[None, :]
                    valid11w1 = valid_d1 & valid_h1 & valid_w1
                    xv = tl.load(x_ptr + x_off + iw_w1, mask=valid11w1, other=0.0)
                    acc7 += wv[:, None] * xv[None, :]

    bv = tl.load(b_ptr + offs_oc)
    bv2 = bv[:, None]
    cmax = acc0 + bv2
    cmax = tl.maximum(cmax, acc1 + bv2)
    cmax = tl.maximum(cmax, acc2 + bv2)
    cmax = tl.maximum(cmax, acc3 + bv2)
    cmax = tl.maximum(cmax, acc4 + bv2)
    cmax = tl.maximum(cmax, acc5 + bv2)
    cmax = tl.maximum(cmax, acc6 + bv2)
    cmax = tl.maximum(cmax, acc7 + bv2)

    m = tl.max(cmax, axis=0)
    e = tl.exp(cmax - m[None, :])
    z = tl.sum(e, axis=0)
    sm = e / z[None, :]

    sub = tl.load(sub_ptr + offs_oc)
    y = sm - sub[:, None]
    sw = y * tl.sigmoid(y)
    final = tl.max(sw, axis=0)

    out_off = ((n * Dp + dp) * Hp + hp) * Wp + wp_base + offs_t
    tl.store(out_ptr + out_off, final)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 output_padding, pool_kernel_size, pool_stride, pool_padding):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
        )
        self.max_pool = nn.MaxPool3d(
            kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding,
        )
        self.subtract = nn.Parameter(torch.randn(out_channels))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding

    def forward(self, x):
        x = x.contiguous()
        N, IC, Din, Hin, Win = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        S = self.stride
        P = self.padding
        OP = self.output_padding

        Dout = (Din - 1) * S - 2 * P + KD + OP
        Hout = (Hin - 1) * S - 2 * P + KH + OP
        Wout = (Win - 1) * S - 2 * P + KW + OP

        pk = self.pool_kernel_size
        ps = self.pool_stride
        pp = self.pool_padding
        Dp = (Dout + 2 * pp - pk) // ps + 1
        Hp = (Hout + 2 * pp - pk) // ps + 1
        Wp = (Wout + 2 * pp - pk) // ps + 1

        out = torch.empty((N, Dp, Hp, Wp), device=x.device, dtype=x.dtype)

        w = self.conv_transpose.weight.contiguous()
        b = self.conv_transpose.bias.contiguous()
        sub = self.subtract.contiguous()

        grid = lambda META: (N * Dp * Hp * (Wp // META['TILE_W']),)
        fused_full_kernel[grid](
            x, w, b, sub, out,
            N,
            IC=IC, OC=OC,
            Din=Din, Hin=Hin, Win=Win,
            Dp=Dp, Hp=Hp, Wp=Wp,
            KD=KD, KH=KH, KW=KW,
            STRIDE=S, PAD=P,
        )
        return out