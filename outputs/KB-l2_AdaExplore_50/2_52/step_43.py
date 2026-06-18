import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def _conv_configs():
    configs = [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ]
    return configs


@triton.autotune(configs=_conv_configs(), key=['M', 'NN', 'K'])
@triton.jit
def _conv2d_mish_bn_kernel(
    x_ptr,           # NHWC: (N, H, W, IC)
    w_ptr,           # (OC, IC*KH*KW) reshaped, but we use (K, OC) layout below
    b_ptr,
    scale_ptr,
    shift_ptr,
    out_ptr,         # NHWC: (N, OH, OW, OC)
    N, IC, H, W, OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    M, NN, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    ohw = OH * OW
    n_idx = offs_m // ohw
    rem_m = offs_m - n_idx * ohw
    oh_idx = rem_m // OW
    ow_idx = rem_m - oh_idx * OW

    m_mask = offs_m < M
    n_mask = offs_n < NN

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate (kh, kw) outer, ic inner so that BLOCK_K runs along contiguous IC.
    base_n = n_idx[:, None] * (H * W * IC)
    for kh in range(0, KH):
        for kw in range(0, KW):
            ih = oh_idx + kh
            iw = ow_idx + kw
            row_base = base_n + ih[:, None] * (W * IC) + iw[:, None] * IC  # [BM, 1]
            kpos_base = (kh * KW + kw) * IC
            for ic_start in range(0, IC, BLOCK_K):
                ic_offs = ic_start + offs_k
                ic_mask = ic_offs < IC

                x_offs = row_base + ic_offs[None, :]
                x_mask = m_mask[:, None] & ic_mask[None, :]
                x_tile = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

                k_offs = kpos_base + ic_offs
                w_offs = k_offs[:, None] * OC + offs_n[None, :]
                w_mask = ic_mask[:, None] & n_mask[None, :]
                w_tile = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

                acc += tl.dot(x_tile, w_tile)

    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    # Mish: x * tanh(softplus(x))
    sp = tl.log(1.0 + tl.exp(acc))
    th = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    y = acc * th

    scale = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0)
    shift = tl.load(shift_ptr + offs_n, mask=n_mask, other=0.0)
    y = y * scale[None, :] + shift[None, :]

    out_offs = (n_idx[:, None] * (OH * OW * OC) +
                oh_idx[:, None] * (OW * OC) +
                ow_idx[:, None] * OC +
                offs_n[None, :])
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_offs, y, mask=out_mask)


def conv2d_mish_bn(x_nhwc, w_kox_oc, bias, scale, shift, N, H, W, IC, OC, OH, OW, KH, KW):
    M = N * OH * OW
    NN = OC
    K = IC * KH * KW
    out = torch.empty((N, OH, OW, OC), device=x_nhwc.device, dtype=x_nhwc.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(NN, meta['BLOCK_N']))
    _conv2d_mish_bn_kernel[grid](
        x_nhwc, w_kox_oc, bias, scale, shift, out,
        N, IC, H, W, OC, OH, OW,
        KH, KW,
        M, NN, K,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.eps = eps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self._cached = None

    def _build_cache(self):
        OC = self.out_channels
        IC = self.in_channels
        KH = KW = self.kernel_size
        # Kernel iterates K as (kh, kw, ic). Reshape weight (OC, IC, KH, KW) -> (OC, KH, KW, IC) -> (K, OC).
        w = self.conv.weight.detach()
        w_perm = w.permute(2, 3, 1, 0).contiguous().view(KH * KW * IC, OC).contiguous()
        bias = self.conv.bias.detach().contiguous()
        rm = self.bn.running_mean
        rv = self.bn.running_var
        bw = self.bn.weight
        bb = self.bn.bias
        invstd = torch.rsqrt(rv + self.eps)
        scale = (bw * invstd).detach().contiguous()
        shift = (bb - rm * bw * invstd).detach().contiguous()
        self._cached = (w_perm, bias, scale, shift)

    def forward(self, x):
        if self.training:
            self._cached = None
            x = self.conv(x)
            x = torch.multiply(torch.tanh(F.softplus(x)), x)
            x = self.bn(x)
            return x
        else:
            N, IC, H, W = x.shape
            KH = KW = self.kernel_size
            OH = H - KH + 1
            OW = W - KW + 1
            OC = self.out_channels

            if self._cached is None:
                self._build_cache()
            w_k_oc, bias, scale, shift = self._cached

            x_nhwc = x.permute(0, 2, 3, 1).contiguous()

            out_nhwc = conv2d_mish_bn(x_nhwc, w_k_oc, bias, scale, shift,
                                      N, H, W, IC, OC, OH, OW, KH, KW)
            out = out_nhwc.permute(0, 3, 1, 2).contiguous()
            return out