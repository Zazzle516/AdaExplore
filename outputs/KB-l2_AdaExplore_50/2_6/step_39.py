import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_softmax_pool_kernel(
    x_ptr,        # input: (N, IC, D_in, H_in, W_in)
    w_ptr,        # weight packed: (IC*KD*KH*KW, OC_padded)
    b_ptr,        # bias: (OC_padded,)
    out_ptr,      # output: (N, OC, Dp, Hp, Wp)
    N, IC, D_in, H_in, W_in,
    OC, Dp, Hp, Wp,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    CUBE: tl.constexpr,  # POOL + KD - 1
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

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    NEG_INF = float('-inf')
    max_acc = tl.full([BLOCK_OC], NEG_INF, dtype=tl.float32)

    d_base = dp * POOL
    h_base = hp * POOL
    w_base = wp * POOL

    x_per_n = IC * D_in * H_in * W_in
    x_per_ic = D_in * H_in * W_in

    # Iterate POOL^3 output positions
    for di in tl.static_range(0, POOL):
        for hi in tl.static_range(0, POOL):
            for wi in tl.static_range(0, POOL):
                d_out = d_base + di
                h_out = h_base + hi
                w_out = w_base + wi

                acc = tl.zeros([BLOCK_OC], dtype=tl.float32)

                # Loop over IC * KD * KH * KW; weight row index = ic*KD*KH*KW + kd*KH*KW + kh*KW + kw
                for ic in tl.static_range(0, 3):
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
                                k_row = (ic * KD * KH * KW
                                         + kd * KH * KW
                                         + kh * KW
                                         + kw)
                                w_ptrs = w_ptr + k_row * BLOCK_OC + oc_offs
                                wv = tl.load(w_ptrs, mask=oc_mask, other=0.0)
                                acc += xv * wv

                acc = acc + bias
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


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


def fused_conv_softmax_pool(x, weight_packed, bias_padded, OC, pool_kernel_size, kernel_size, BLOCK_OC):
    N, IC, D_in, H_in, W_in = x.shape
    KD = KH = KW = kernel_size
    D_out = D_in - KD + 1
    H_out = H_in - KH + 1
    W_out = W_in - KW + 1
    POOL = pool_kernel_size * pool_kernel_size
    Dp = D_out // POOL
    Hp = H_out // POOL
    Wp = W_out // POOL

    x = x.contiguous()
    out = torch.empty((N, OC, Dp, Hp, Wp), device=x.device, dtype=x.dtype)

    grid = (N * Dp * Hp * Wp,)
    fused_conv_softmax_pool_kernel[grid](
        x, weight_packed, bias_padded, out,
        N, IC, D_in, H_in, W_in,
        OC, Dp, Hp, Wp,
        KD=KD, KH=KH, KW=KW,
        POOL=POOL,
        BLOCK_OC=BLOCK_OC,
        CUBE=POOL + KD - 1,
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

        self.BLOCK_OC = max(_next_pow2(out_channels), 16)
        self._packed_dirty = True

    def _pack_weights(self):
        # weight shape: (OC, IC, KD, KH, KW)
        w = self.conv.weight.detach()
        OC, IC, KD, KH, KW = w.shape
        K = IC * KD * KH * KW
        # reshape to (OC, K), then transpose to (K, OC), pad OC to BLOCK_OC
        wp = w.reshape(OC, K).t().contiguous()  # (K, OC)
        if self.BLOCK_OC > OC:
            pad = torch.zeros((K, self.BLOCK_OC - OC), device=w.device, dtype=w.dtype)
            wp = torch.cat([wp, pad], dim=1).contiguous()
        b = self.conv.bias.detach()
        if self.BLOCK_OC > OC:
            bpad = torch.zeros((self.BLOCK_OC - OC,), device=b.device, dtype=b.dtype)
            bp = torch.cat([b, bpad], dim=0).contiguous()
        else:
            bp = b.contiguous()
        self._weight_packed = wp
        self._bias_padded = bp
        self._packed_dirty = False

    def forward(self, x):
        POOL = self.pool_kernel_size * self.pool_kernel_size
        N, IC, D, H, W = x.shape
        D_out = D - self.kernel_size + 1
        H_out = H - self.kernel_size + 1
        W_out = W - self.kernel_size + 1

        if (D_out % POOL == 0 and H_out % POOL == 0 and W_out % POOL == 0
                and IC == 3 and not self.training):
            if self._packed_dirty or not hasattr(self, '_weight_packed') or self._weight_packed.device != x.device:
                self._pack_weights()
                if self._weight_packed.device != x.device:
                    self._weight_packed = self._weight_packed.to(x.device)
                    self._bias_padded = self._bias_padded.to(x.device)
            return fused_conv_softmax_pool(
                x, self._weight_packed, self._bias_padded,
                self.out_channels, self.pool_kernel_size, self.kernel_size,
                self.BLOCK_OC)
        else:
            x = self.conv(x)
            x = torch.softmax(x, dim=1)
            x = F.max_pool3d(x, self.pool_kernel_size)
            x = F.max_pool3d(x, self.pool_kernel_size)
            return x

    def train(self, mode=True):
        self._packed_dirty = True
        return super().train(mode)