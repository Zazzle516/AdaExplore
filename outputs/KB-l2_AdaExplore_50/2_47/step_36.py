import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'IC'],
)
@triton.jit
def conv3d_nhwc_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    M, N_,
    OHW, OW_, IHW, IW_,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # = IC (channels block; we assume IC fits)
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)  # IC

    m_mask = offs_m < M
    n_mask = offs_n < N_

    # decode m -> (n_idx, od, oh, ow)
    n_idx = offs_m // (OD * OHW)
    rem = offs_m % (OD * OHW)
    od = rem // OHW
    rem2 = rem % OHW
    oh = rem2 // OW_
    ow = rem2 % OW_

    # x is NDHWC: x[n, d, h, w, ic] with strides (ID*IHW*IC, IHW*IC, IW_*IC, IC, 1)
    # base offset (top-left of receptive field, channel 0)
    x_base = (n_idx * (ID * IHW * IC)
              + od * (IHW * IC)
              + oh * (IW_ * IC)
              + ow * IC)  # [BLOCK_M]

    k_mask = offs_k < IC  # [BLOCK_K]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # w reshaped to [KD*KH*KW*IC, OC]
    # weight layout: w[kd, kh, kw, ic, oc], strides ((KH*KW*IC*OC), (KW*IC*OC), (IC*OC), OC, 1)
    KHW_IC_OC = KH * KW * IC * OC
    KW_IC_OC = KW * IC * OC
    IC_OC = IC * OC

    for kd in tl.static_range(0, KD):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                # x offsets [BLOCK_M, BLOCK_K]
                x_off = (x_base[:, None]
                         + kd * (IHW * IC)
                         + kh * (IW_ * IC)
                         + kw * IC
                         + offs_k[None, :])
                x_mask = m_mask[:, None] & k_mask[None, :]
                x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                # w offsets [BLOCK_K, BLOCK_N]
                w_off = (kd * KHW_IC_OC
                         + kh * KW_IC_OC
                         + kw * IC_OC
                         + offs_k[:, None] * OC
                         + offs_n[None, :])
                w_mask = k_mask[:, None] & n_mask[None, :]
                w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals)

    b = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + b[None, :]

    # Mish: x * tanh(softplus(x))
    sp = tl.log(1.0 + tl.exp(acc))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    mish = acc * th
    e2m = tl.exp(2.0 * mish)
    out_val = (e2m - 1.0) / (e2m + 1.0)

    # out is NDHWC: out[n, d, h, w, oc]
    out_off = (n_idx[:, None] * (OD * OHW * OC)
               + od[:, None] * (OHW * OC)
               + oh[:, None] * (OW_ * OC)
               + ow[:, None] * OC
               + offs_n[None, :])
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_off, out_val, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kd = self.kh = self.kw = kernel_size
        else:
            self.kd, self.kh, self.kw = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)

        # Pre-permute weight to (KD, KH, KW, IC, OC) for NDHWC layout
        # Will be cached lazily on first forward call
        self._w_nhwc = None
        self._b_contig = None

    def _get_weight_nhwc(self):
        if self._w_nhwc is None or self._w_nhwc.device != self.conv.weight.device:
            # conv.weight is (OC, IC, KD, KH, KW)
            w = self.conv.weight.detach()
            # -> (KD, KH, KW, IC, OC)
            w_nhwc = w.permute(2, 3, 4, 1, 0).contiguous()
            self._w_nhwc = w_nhwc
            self._b_contig = self.conv.bias.detach().contiguous()
        return self._w_nhwc, self._b_contig

    def forward(self, x):
        if (self.stride == (1, 1, 1) and self.padding == (0, 0, 0)
                and x.is_cuda and x.dtype == torch.float32):
            N, IC, ID, IH, IW = x.shape
            OC = self.out_channels
            KD, KH, KW = self.kd, self.kh, self.kw
            OD = ID - KD + 1
            OH = IH - KH + 1
            OW = IW - KW + 1

            # Convert input NCDHW -> NDHWC
            x_nhwc = x.permute(0, 2, 3, 4, 1).contiguous()
            w_nhwc, b = self._get_weight_nhwc()

            out_nhwc = torch.empty((N, OD, OH, OW, OC), device=x.device, dtype=x.dtype)

            M = N * OD * OH * OW
            N_ = OC
            OHW = OH * OW
            IHW = IH * IW

            # BLOCK_K is the IC dimension; needs to be a power of two >= IC for tl.dot
            def next_pow2(x):
                p = 1
                while p < x:
                    p *= 2
                return p

            BLOCK_K = max(16, next_pow2(IC))

            grid = lambda meta: (
                triton.cdiv(M, meta['BLOCK_M']),
                triton.cdiv(N_, meta['BLOCK_N']),
            )

            conv3d_nhwc_kernel[grid](
                x_nhwc, w_nhwc, b, out_nhwc,
                N, IC, ID, IH, IW,
                OC, OD, OH, OW,
                M, N_,
                OHW, OW, IHW, IW,
                KD, KH, KW,
                BLOCK_K=BLOCK_K,
            )

            # NDHWC -> NCDHW
            out = out_nhwc.permute(0, 4, 1, 2, 3).contiguous()
            return out

        # fallback
        x = self.conv(x)
        x = F.mish(x)
        x = torch.tanh(x)
        return x