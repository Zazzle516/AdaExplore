import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 8}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_W': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_W': 32}, num_warps=8, num_stages=3),
    ],
    key=['N', 'Dp', 'Hp', 'Wp'],
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
    Dout: tl.constexpr, Hout: tl.constexpr, Wout: tl.constexpr,
    Dp: tl.constexpr, Hp: tl.constexpr, Wp: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # one program handles BLOCK_W consecutive wp positions for a given (n, dp, hp)
    pid = tl.program_id(0)
    W_tiles = (Wp + BLOCK_W - 1) // BLOCK_W
    HpWt = Hp * W_tiles
    DpHpWt = Dp * HpWt

    n = pid // DpHpWt
    rem = pid % DpHpWt
    dp = rem // HpWt
    rem2 = rem % HpWt
    hp = rem2 // W_tiles
    wt = rem2 % W_tiles

    offs_w = wt * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = offs_w < Wp

    d0 = 2 * dp
    h0 = 2 * hp

    offs_oc = tl.arange(0, OC)  # 16

    KDHKW = KD * KH * KW

    neg_inf = float('-inf')
    cmax = tl.full((OC, BLOCK_W), neg_inf, dtype=tl.float32)

    bv = tl.load(b_ptr + offs_oc)  # (OC,)

    # With STRIDE=2, PAD=1, KD=KH=KW=3:
    # For each output pos p (= 2*pp + dpp, where dpp in {0,1}):
    #   num = p + 1 - k = (2*pp + dpp) + 1 - k
    #   valid iff num % 2 == 0  =>  (dpp + 1 - k) even => k has same parity as (dpp+1)
    #   dpp=0 -> k in {1} (odd) -> only k=1 valid (single tap)
    #   dpp=1 -> k in {0, 2} (even) -> two taps
    #   id_ = num/2
    # We can precompute which kernel taps are valid per dpp.
    # But static_range over KD with parity check would still issue masked loads.
    # Better: enumerate only valid taps.

    # We'll unroll: for each (ddp, dhp, dwp) in {0,1}^3, compute the conv-transpose
    # output and update cmax.

    for ddp in tl.static_range(0, 2):
        d_out = d0 + ddp
        # valid kd values: those with (d_out + PAD - kd) % STRIDE == 0
        # For STRIDE=2, PAD=1, KD=3: d_out parity decides which kd taps are valid.
        # We still iterate kd in [0,KD); inside, check validity. But we know exactly
        # one or two will be valid. Keep static_range for compile-time unrolling.
        for dhp in tl.static_range(0, 2):
            h_out = h0 + dhp

            for dwp in tl.static_range(0, 2):
                w_out = 2 * offs_w + dwp  # (BLOCK_W,)

                acc = tl.zeros((OC, BLOCK_W), dtype=tl.float32)

                for kd in tl.static_range(0, KD):
                    num_d = d_out + PAD - kd
                    # only proceed if divisible; for non-divisible, skip via guard
                    rem_d = num_d % STRIDE
                    id_ = num_d // STRIDE
                    valid_d = (rem_d == 0) & (id_ >= 0) & (id_ < Din)
                    for kh in tl.static_range(0, KH):
                        num_h = h_out + PAD - kh
                        rem_h = num_h % STRIDE
                        ih_ = num_h // STRIDE
                        valid_h = (rem_h == 0) & (ih_ >= 0) & (ih_ < Hin)
                        valid_dh = valid_d & valid_h
                        for kw in tl.static_range(0, KW):
                            num_w = w_out + PAD - kw  # (BLOCK_W,)
                            rem_w = num_w % STRIDE
                            iw_ = num_w // STRIDE
                            valid_w = (rem_w == 0) & (iw_ >= 0) & (iw_ < Win)
                            valid = valid_dh & valid_w & mask_w

                            for ic in tl.static_range(0, IC):
                                x_off = (((n * IC + ic) * Din + id_) * Hin + ih_) * Win + iw_
                                xv = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                                w_base = (((ic * OC) * KD + kd) * KH + kh) * KW + kw
                                w_off = w_base + offs_oc * KDHKW
                                wv = tl.load(w_ptr + w_off)  # (OC,)
                                acc += wv[:, None] * xv[None, :]

                acc = acc + bv[:, None]
                cmax = tl.maximum(cmax, acc)

    # softmax over channels (axis=0)
    m = tl.max(cmax, axis=0)
    e = tl.exp(cmax - m[None, :])
    z = tl.sum(e, axis=0)
    sm = e / z[None, :]

    sub = tl.load(sub_ptr + offs_oc)
    y = sm - sub[:, None]
    sw = y * tl.sigmoid(y)
    final = tl.max(sw, axis=0)

    out_off = ((n * Dp + dp) * Hp + hp) * Wp + offs_w
    tl.store(out_ptr + out_off, final, mask=mask_w)


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

        grid = lambda META: (N * Dp * Hp * triton.cdiv(Wp, META['BLOCK_W']),)
        fused_full_kernel[grid](
            x, w, b, sub, out,
            N,
            IC=IC, OC=OC,
            Din=Din, Hin=Hin, Win=Win,
            Dout=Dout, Hout=Hout, Wout=Wout,
            Dp=Dp, Hp=Hp, Wp=Wp,
            KD=KD, KH=KH, KW=KW,
            STRIDE=S, PAD=P,
        )
        return out