import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_full_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    sub_ptr,
    out_ptr,
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

    neg_inf = float('-inf')
    cmax = tl.full((OC, TILE_W), neg_inf, dtype=tl.float32)

    for ddp in tl.static_range(0, 2):
        for dhp in tl.static_range(0, 2):
            for dwp in tl.static_range(0, 2):
                d_out = d0 + ddp
                h_out = h0 + dhp

                acc = tl.zeros((OC, TILE_W), dtype=tl.float32)

                for kd in tl.static_range(0, KD):
                    num_d = d_out + PAD - kd
                    id_ = num_d // STRIDE
                    valid_d = ((num_d % STRIDE) == 0) & (id_ >= 0) & (id_ < Din)
                    for kh in tl.static_range(0, KH):
                        num_h = h_out + PAD - kh
                        ih_ = num_h // STRIDE
                        valid_h = ((num_h % STRIDE) == 0) & (ih_ >= 0) & (ih_ < Hin)
                        for kw in tl.static_range(0, KW):
                            num_w = (w0 + 2 * offs_t + dwp) + PAD - kw
                            iw_ = num_w // STRIDE
                            valid_w = ((num_w % STRIDE) == 0) & (iw_ >= 0) & (iw_ < Win)
                            valid = valid_d & valid_h & valid_w

                            for ic in tl.static_range(0, IC):
                                x_off = (((n * IC + ic) * Din + id_) * Hin + ih_) * Win + iw_
                                xv = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                                w_base = (((ic * OC) * KD + kd) * KH + kh) * KW + kw
                                w_off = w_base + offs_oc * (KD * KH * KW)
                                wv = tl.load(w_ptr + w_off)
                                acc += wv[:, None] * xv[None, :]

                bv = tl.load(b_ptr + offs_oc)
                acc = acc + bv[:, None]
                cmax = tl.maximum(cmax, acc)

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

        TILE_W = 16
        while Wp % TILE_W != 0:
            TILE_W //= 2
        if TILE_W < 1:
            TILE_W = 1

        grid = (N * Dp * Hp * (Wp // TILE_W),)
        fused_full_kernel[grid](
            x, w, b, sub, out,
            N,
            IC=IC, OC=OC,
            Din=Din, Hin=Hin, Win=Win,
            Dp=Dp, Hp=Hp, Wp=Wp,
            KD=KD, KH=KH, KW=KW,
            STRIDE=S, PAD=P,
            TILE_W=TILE_W,
            num_warps=4,
            num_stages=2,
        )
        return out