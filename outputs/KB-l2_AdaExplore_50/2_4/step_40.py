import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_mish2_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    M, K_total,
    # strides for x (NHWC): n, h, w, c
    sx_n, sx_h, sx_w,
    # strides for w (OC, KH, KW, IC) contiguous
    # strides for out (NHWC): n, h, w, c
    so_n, so_h, so_w,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = N*OH*OW, N_dim = OC, K_total = KH*KW*IC
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # output spatial
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output channels
    offs_k = tl.arange(0, BLOCK_K)

    # Decompose offs_m into (n, oh, ow)
    ow = offs_m % OW
    tmp = offs_m // OW
    oh = tmp % OH
    n_idx = tmp // OH

    m_mask = offs_m < M
    n_mask = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # base x pointer per row
    # x[n, oh+kh, ow+kw, ic]; with stride 1, padding 0
    x_base = n_idx * sx_n + oh * sx_h + ow * sx_w  # [BLOCK_M]

    KK = KH * KW * IC

    for k_start in range(0, KK, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]
        k_mask = k_idx < KK

        ic = k_idx % IC
        tmp_k = k_idx // IC
        kw_i = tmp_k % KW
        kh_i = tmp_k // KW

        # x offsets: x_base[:, None] + kh_i*sx_h + kw_i*sx_w + ic
        x_off = x_base[:, None] + kh_i[None, :] * sx_h + kw_i[None, :] * sx_w + ic[None, :]
        x_mask = m_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

        # w[oc, kh, kw, ic] -> w_ptr + oc*KK + k_idx
        w_off = offs_n[None, :] * KK + k_idx[:, None]
        w_mask = n_mask[None, :] & k_mask[:, None]
        w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    # add bias
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    # double mish
    # mish(a) = a * tanh(softplus(a))
    # softplus = log(1+exp(a))
    sp1 = tl.log(1.0 + tl.exp(acc))
    e1 = tl.exp(2.0 * sp1)
    t1 = (e1 - 1.0) / (e1 + 1.0)
    y = acc * t1
    sp2 = tl.log(1.0 + tl.exp(y))
    e2 = tl.exp(2.0 * sp2)
    t2 = (e2 - 1.0) / (e2 + 1.0)
    z = y * t2

    # store NHWC
    out_off = n_idx[:, None] * so_n + oh[:, None] * so_h + ow[:, None] * so_w + offs_n[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_off, z, mask=out_mask)


def conv_mish2(x, weight, bias):
    # x: (N, IC, IH, IW) -> NHWC
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = weight.shape
    OH = IH - KH + 1
    OW = IW - KW + 1

    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    # weight (OC, IC, KH, KW) -> (OC, KH, KW, IC) contiguous
    w_perm = weight.permute(0, 2, 3, 1).contiguous()

    out_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

    M = N * OH * OW
    K_total = KH * KW * IC

    sx_n = IH * IW * IC
    sx_h = IW * IC
    sx_w = IC

    so_n = OH * OW * OC
    so_h = OW * OC
    so_w = OC

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(OC, META['BLOCK_N']),
    )

    conv_mish2_kernel[grid](
        x_nhwc, w_perm, bias, out_nhwc,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        M, K_total,
        sx_n, sx_h, sx_w,
        so_n, so_h, so_w,
    )

    out = out_nhwc.permute(0, 3, 1, 2).contiguous()
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return conv_mish2(x, self.conv.weight, self.conv.bias)