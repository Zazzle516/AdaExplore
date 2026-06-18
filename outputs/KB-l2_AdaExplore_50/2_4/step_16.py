import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_mish_mish_kernel(
    x_ptr,           # [N, H, W, IC] NHWC
    w_ptr,           # [OC, KH, KW, IC]
    b_ptr,           # [OC]
    out_ptr,         # [N, OH, OW, OC] NHWC
    N, IC, H, W,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr,   # tile over OH*OW
    BLOCK_N: tl.constexpr,   # tile over OC
    BLOCK_K: tl.constexpr,   # tile over IC
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # spatial idx
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # OC idx
    offs_k = tl.arange(0, BLOCK_K)                     # IC idx

    # decompose offs_m into (oh, ow)
    oh = offs_m // OW
    ow = offs_m % OW
    m_mask = offs_m < (OH * OW)
    n_mask = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # base pointer for this batch in input
    x_batch_ptr = x_ptr + pid_b * H * W * IC

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh  # no padding
            iw = ow + kw
            # x: [BLOCK_M, BLOCK_K]
            x_offs = (ih[:, None] * W + iw[:, None]) * IC + offs_k[None, :]
            x_mask = m_mask[:, None] & (offs_k[None, :] < IC)
            x_tile = tl.load(x_batch_ptr + x_offs, mask=x_mask, other=0.0)

            # w: [BLOCK_K, BLOCK_N] -> weight layout [OC, KH, KW, IC]
            w_offs = offs_n[None, :] * (KH * KW * IC) + (kh * KW + kw) * IC + offs_k[:, None]
            w_mask = n_mask[None, :] & (offs_k[:, None] < IC)
            w_tile = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

            acc += tl.dot(x_tile, w_tile)

    # bias
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    # double mish: y = x * tanh(softplus(x)), apply twice
    # softplus stable: where x > 20, x, log1p(exp(x))
    x1 = acc
    sp1 = tl.where(x1 > 20.0, x1, tl.log(1.0 + tl.exp(x1)))
    e1 = tl.exp(2.0 * sp1)
    t1 = (e1 - 1.0) / (e1 + 1.0)
    y = x1 * t1
    sp2 = tl.where(y > 20.0, y, tl.log(1.0 + tl.exp(y)))
    e2 = tl.exp(2.0 * sp2)
    t2 = (e2 - 1.0) / (e2 + 1.0)
    z = y * t2

    # store: out [N, OH, OW, OC]
    out_batch_ptr = out_ptr + pid_b * OH * OW * OC
    out_offs = offs_m[:, None] * OC + offs_n[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_batch_ptr + out_offs, z, mask=out_mask)


def conv2d_mish_mish(x_nhwc, w_oc_khkw_ic, bias, OH, OW, KH, KW):
    N, H, W, IC = x_nhwc.shape
    OC = w_oc_khkw_ic.shape[0]

    out = torch.empty((N, OH, OW, OC), device=x_nhwc.device, dtype=x_nhwc.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64  # IC=64 fits exactly

    grid = (triton.cdiv(OH * OW, BLOCK_M), triton.cdiv(OC, BLOCK_N), N)

    conv2d_mish_mish_kernel[grid](
        x_nhwc, w_oc_khkw_ic, bias, out,
        N, IC, H, W, OC, OH, OW,
        KH=KH, KW=KW,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        # x: [N, IC, H, W]
        N, IC, H, W = x.shape
        KH = KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1

        # to NHWC contiguous
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        # weight: [OC, IC, KH, KW] -> [OC, KH, KW, IC]
        w = self.conv.weight.permute(0, 2, 3, 1).contiguous()
        b = self.conv.bias.contiguous()

        out_nhwc = conv2d_mish_mish(x_nhwc, w, b, OH, OW, KH, KW)
        # back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out