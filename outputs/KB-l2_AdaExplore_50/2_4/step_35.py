import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _mish_inline(x):
    # Stable mish: x * tanh(softplus(x))
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    t = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    return x * t


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_im2col_mish2_kernel(
    x_ptr,       # input (N, IC, IH, IW) contiguous NCHW
    w_ptr,       # weight (OC, IC, KH, KW) contiguous
    b_ptr,       # bias (OC,)
    out_ptr,     # output (N, OC, OH, OW) contiguous NCHW
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    M, K_, NN,   # M = OH*OW, K = IC*KH*KW, NN = OC
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)     # batch index
    pid_oc = tl.program_id(1)    # OC tile
    pid_m = tl.program_id(2)     # M tile (spatial)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # spatial indices
    offs_oc = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N) # output channel indices
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    oc_mask = offs_oc < NN

    # decompose spatial: oh, ow
    oh = offs_m // OW
    ow = offs_m % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # input/output base for this batch
    x_batch = x_ptr + pid_n * (IC * IH * IW)

    for k_start in range(0, K_, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]
        k_mask = k_idx < K_

        # decompose k -> (ic, kh, kw)
        kw_i = k_idx % KW
        khc = k_idx // KW
        kh_i = khc % KH
        ic_i = khc // KH

        # input spatial positions: ih = oh + kh_i, iw = ow + kw_i (no padding)
        # X block: [BLOCK_M, BLOCK_K]
        ih = oh[:, None] + kh_i[None, :]
        iw = ow[:, None] + kw_i[None, :]
        ic_b = ic_i[None, :]  # broadcasts to [1, BLOCK_K]

        x_offsets = ic_b * (IH * IW) + ih * IW + iw  # [BLOCK_M, BLOCK_K]
        x_mask = m_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_batch + x_offsets, mask=x_mask, other=0.0)

        # Weight block: [BLOCK_K, BLOCK_N]
        # weight layout: (OC, IC, KH, KW) -> w[oc, ic, kh, kw] = oc*K + (ic*KH*KW + kh*KW + kw)
        w_offsets = offs_oc[None, :] * K_ + k_idx[:, None]
        w_mask = k_mask[:, None] & oc_mask[None, :]
        w_vals = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    # Add bias
    bias = tl.load(b_ptr + offs_oc, mask=oc_mask, other=0.0)
    acc = acc + bias[None, :]

    # Two mish activations fused
    acc = _mish_inline(acc)
    acc = _mish_inline(acc)

    # Store: out[pid_n, oc, oh, ow]
    # output stride: N, OC, OH, OW -> contiguous
    out_batch = out_ptr + pid_n * (OC * OH * OW)
    out_offsets = offs_oc[None, :] * (OH * OW) + offs_m[:, None]
    out_mask = m_mask[:, None] & oc_mask[None, :]
    tl.store(out_batch + out_offsets, acc, mask=out_mask)


def conv_mish2(x, weight, bias):
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = weight.shape
    OH = IH - KH + 1
    OW = IW - KW + 1

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    M = OH * OW
    K_ = IC * KH * KW
    NN = OC

    def grid(meta):
        return (
            N,
            triton.cdiv(NN, meta['BLOCK_N']),
            triton.cdiv(M, meta['BLOCK_M']),
        )

    conv_im2col_mish2_kernel[grid](
        x, weight, bias, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        M, K_, NN,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return conv_mish2(x, self.conv.weight, self.conv.bias)