import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Fused ConvTranspose3d + MaxPool3d + Softmax(channel) + sub + swish + channel-max
# Architecture-specific values:
#   in_channels=3, out_channels=16, kernel=3, stride=2, padding=1, output_padding=1
#   D_in=16, H_in=32, W_in=32 -> D_out=32, H_out=64, W_out=64
#   pool: k=2, s=2, p=0 -> Dp=16, Hp=32, Wp=32
#
# Strategy: one program computes one (n, dp, hp, wp) output element.
# It computes the conv_transpose output for all 16 channels at the 8 pre-pool
# positions inside that pool window (in registers), then performs:
#   - per-channel max over the 8 positions  -> length-16 vector
#   - softmax across 16 channels
#   - subtract per-channel param
#   - swish (y * sigmoid(y))
#   - max across channels -> scalar output


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
    Dout: tl.constexpr, Hout: tl.constexpr, Wout: tl.constexpr,
    Dp: tl.constexpr, Hp: tl.constexpr, Wp: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
):
    pid = tl.program_id(0)
    # decompose pid -> (n, dp, hp, wp)
    HpWp = Hp * Wp
    DpHpWp = Dp * HpWp
    n = pid // DpHpWp
    rem = pid % DpHpWp
    dp = rem // HpWp
    rem2 = rem % HpWp
    hp = rem2 // Wp
    wp = rem2 % Wp

    # Pool window in conv-transpose output coords:
    #   d_out in [2*dp, 2*dp+1], h_out in [2*hp, 2*hp+1], w_out in [2*wp, 2*wp+1]
    d0 = 2 * dp
    h0 = 2 * hp
    w0 = 2 * wp

    offs_oc = tl.arange(0, OC)  # 16

    # Accumulate per-channel max over the 8 spatial positions
    neg_inf = float('-inf')
    cmax = tl.full((OC,), neg_inf, dtype=tl.float32)

    # Preload entire weight tensor once: shape (IC, OC, KD, KH, KW) flattened.
    # We index as w_cache[ic, kd, kh, kw, oc] via per-(ic,kd,kh,kw) slice of length OC.
    bv = tl.load(b_ptr + offs_oc)

    # Loop over the 8 positions in pool window (unrolled by Triton since constexpr)
    for ddp in tl.static_range(0, 2):
        for dhp in tl.static_range(0, 2):
            for dwp in tl.static_range(0, 2):
                d_out = d0 + ddp
                h_out = h0 + dhp
                w_out = w0 + dwp

                # accumulator for all OC at this output spatial location
                acc = tl.zeros((OC,), dtype=tl.float32)

                # Loop over kernel taps; for each (kd,kh,kw) check if maps to valid input
                # id = (d_out + PAD - kd) / STRIDE  if divisible
                for kd in tl.static_range(0, KD):
                    num_d = d_out + PAD - kd
                    id_ = num_d // STRIDE
                    valid_d = ((num_d % STRIDE) == 0) & (id_ >= 0) & (id_ < Din)
                    for kh in tl.static_range(0, KH):
                        num_h = h_out + PAD - kh
                        ih_ = num_h // STRIDE
                        valid_h = ((num_h % STRIDE) == 0) & (ih_ >= 0) & (ih_ < Hin)
                        for kw in tl.static_range(0, KW):
                            num_w = w_out + PAD - kw
                            iw_ = num_w // STRIDE
                            valid_w = ((num_w % STRIDE) == 0) & (iw_ >= 0) & (iw_ < Win)
                            valid = valid_d & valid_h & valid_w

                            for ic in tl.static_range(0, IC):
                                x_off = (((n * IC + ic) * Din + id_) * Hin + ih_) * Win + iw_
                                xv = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                                w_base = (((ic * OC) * KD + kd) * KH + kh) * KW + kw
                                w_off = w_base + offs_oc * (KD * KH * KW)
                                wv = tl.load(w_ptr + w_off)
                                acc += xv * wv

                # add bias
                acc = acc + bv

                # update channel-wise max over pool window
                cmax = tl.maximum(cmax, acc)

    # cmax is the (16,) post-maxpool vector for this (n, dp, hp, wp).
    # softmax across channels:
    m = tl.max(cmax, axis=0)
    e = tl.exp(cmax - m)
    z = tl.sum(e, axis=0)
    sm = e / z

    sub = tl.load(sub_ptr + offs_oc)
    y = sm - sub
    sw = y * tl.sigmoid(y)
    final = tl.max(sw, axis=0)

    out_off = ((n * Dp + dp) * Hp + hp) * Wp + wp
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

        # pool with k=2, s=2, p=0 (assumed; matches the architecture)
        pk = self.pool_kernel_size
        ps = self.pool_stride
        pp = self.pool_padding
        Dp = (Dout + 2 * pp - pk) // ps + 1
        Hp = (Hout + 2 * pp - pk) // ps + 1
        Wp = (Wout + 2 * pp - pk) // ps + 1

        out = torch.empty((N, Dp, Hp, Wp), device=x.device, dtype=x.dtype)

        w = self.conv_transpose.weight.contiguous()  # (IC, OC, KD, KH, KW)
        b = self.conv_transpose.bias.contiguous()    # (OC,)
        sub = self.subtract.contiguous()

        grid = (N * Dp * Hp * Wp,)
        fused_full_kernel[grid](
            x, w, b, sub, out,
            N,
            IC=IC, OC=OC,
            Din=Din, Hin=Hin, Win=Win,
            Dout=Dout, Hout=Hout, Wout=Wout,
            Dp=Dp, Hp=Hp, Wp=Wp,
            KD=KD, KH=KH, KW=KW,
            STRIDE=S, PAD=P,
            num_warps=4,
            num_stages=2,
        )
        return out