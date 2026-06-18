import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_mish2_nhwc_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, H, W, IC, OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_OC: tl.constexpr, BLOCK_IC: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_n = tl.program_id(2)

    # M dim is OH*OW, tiled
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]

    oh = m_offs // OW
    ow = m_offs % OW
    m_mask = m_offs < (OH * OW)
    oc_mask = oc_offs < OC

    # x layout: [N, H, W, IC] contiguous
    # w layout: [OC, KH, KW, IC] - we'll load as [KH*KW*IC, OC] effectively
    # Output layout: [N, OH, OW, OC]

    acc = tl.zeros((BLOCK_M, BLOCK_OC), dtype=tl.float32)
    # Load bias
    b = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b[None, :]

    n_idx = pid_n
    x_base = n_idx * H * W * IC

    ic_arange = tl.arange(0, BLOCK_IC)

    for kh in tl.static_range(KH):
        ih = oh + kh  # since padding=0
        for kw in tl.static_range(KW):
            iw = ow + kw
            # spatial valid (always valid since OH=H-KH+1, etc., no padding)
            # x offset: x_base + ih*W*IC + iw*IC + ic
            x_row_base = x_base + ih * (W * IC) + iw * IC  # [BLOCK_M]
            for ic_start in range(0, IC, BLOCK_IC):
                ic_offs = ic_start + ic_arange
                ic_mask = ic_offs < IC

                # x: [BLOCK_M, BLOCK_IC]
                x_ptrs = x_row_base[:, None] + ic_offs[None, :]
                x_vals = tl.load(
                    x_ptr + x_ptrs,
                    mask=m_mask[:, None] & ic_mask[None, :],
                    other=0.0,
                )

                # w: [OC, KH, KW, IC] -> w[oc, kh, kw, ic]
                # offset = oc*KH*KW*IC + kh*KW*IC + kw*IC + ic
                # We need [BLOCK_IC, BLOCK_OC]
                w_ptrs = (
                    oc_offs[None, :] * (KH * KW * IC)
                    + kh * (KW * IC)
                    + kw * IC
                    + ic_offs[:, None]
                )
                w_vals = tl.load(
                    w_ptr + w_ptrs,
                    mask=oc_mask[None, :] & ic_mask[:, None],
                    other=0.0,
                )

                acc += tl.dot(x_vals, w_vals)

    # Apply double mish
    # mish(x) = x * tanh(softplus(x))
    # softplus(x) = log(1+exp(x))
    sp1 = tl.log(1.0 + tl.exp(acc))
    e1 = tl.exp(2.0 * sp1)
    t1 = (e1 - 1.0) / (e1 + 1.0)
    y = acc * t1
    sp2 = tl.log(1.0 + tl.exp(y))
    e2 = tl.exp(2.0 * sp2)
    t2 = (e2 - 1.0) / (e2 + 1.0)
    z = y * t2

    # Store: out [N, OH, OW, OC]
    out_base = n_idx * OH * OW * OC
    out_ptrs = out_base + m_offs[:, None] * OC + oc_offs[None, :]
    tl.store(out_ptrs, z, mask=m_mask[:, None] & oc_mask[None, :])


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # Pre-permute weight to [OC, KH, KW, IC] contiguous
        self._cached_weight = None
        self._cached_bias = None

    def _get_weight_nhwc(self):
        w = self.conv.weight  # [OC, IC, KH, KW]
        w_nhwc = w.permute(0, 2, 3, 1).contiguous()
        return w_nhwc

    def forward(self, x):
        # x: [N, IC, H, W]
        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1

        # Convert input to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        w_nhwc = self._get_weight_nhwc()
        bias = self.conv.bias.contiguous()

        out_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

        BLOCK_M = 64
        BLOCK_OC = 64
        BLOCK_IC = 32

        grid = (
            triton.cdiv(OH * OW, BLOCK_M),
            triton.cdiv(OC, BLOCK_OC),
            N,
        )

        conv_mish2_nhwc_kernel[grid](
            x_nhwc, w_nhwc, bias, out_nhwc,
            N, H, W, IC, OC, OH, OW,
            KH=KH, KW=KW,
            BLOCK_M=BLOCK_M, BLOCK_OC=BLOCK_OC, BLOCK_IC=BLOCK_IC,
            num_warps=4, num_stages=2,
        )

        # Convert back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out