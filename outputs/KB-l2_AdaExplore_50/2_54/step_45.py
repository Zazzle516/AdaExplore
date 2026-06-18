import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    ],
    key=['C_in', 'C_out', 'H_out', 'W_out', 'KH', 'KW'],
)
@triton.jit
def conv2d_fused_nhwc_kernel(
    x_ptr, w_ptr, b_ptr, m_ptr, out_ptr,
    N, C_in, H, W,
    C_out, H_out, W_out,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_HW: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    oh = hw_offs // W_out
    ow = hw_offs % W_out

    oc_mask = oc_offs < C_out
    hw_mask = hw_offs < (H_out * W_out)

    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    # K axis = KH * KW * C_in
    K = KH * KW * C_in
    k_offs = tl.arange(0, BLOCK_K)

    # Input is NCHW: x[n, ic, ih, iw] = n*C_in*H*W + ic*H*W + ih*W + iw
    # Weight in (OC, KH, KW, C_in): w[oc, kh, kw, ic] = oc*KH*KW*C_in + kh*KW*C_in + kw*C_in + ic

    n_base = pid_n * C_in * H * W
    HW = H * W

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + k_offs  # [BLOCK_K]
        k_mask = k_idx < K

        # decompose k_idx -> (kh, kw, ic)
        ic = k_idx % C_in
        khw = k_idx // C_in
        kw = khw % KW
        kh = khw // KW

        # x indices: shape [BLOCK_HW, BLOCK_K] -- read directly from NCHW
        ih = oh[:, None] + kh[None, :]
        iw = ow[:, None] + kw[None, :]
        x_offsets = n_base + ic[None, :] * HW + ih * W + iw
        x_mask = hw_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)  # [BLOCK_HW, BLOCK_K]

        # w indices: shape [BLOCK_K, BLOCK_OC]
        w_offsets = oc_offs[None, :] * K + k_idx[:, None]
        w_mask = k_mask[:, None] & oc_mask[None, :]
        w_tile = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_OC]

        acc += tl.dot(x_tile, w_tile)

    # bias
    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    # multiplier per out channel
    m_vals = tl.load(m_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]

    acc = acc + b_vals[None, :]
    acc = acc * m_vals[None, :]

    # LeakyReLU (default negative_slope=0.01)
    acc = tl.where(acc >= 0, acc, acc * 0.01)

    # GELU tanh approximation: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    k0 = 0.7978845608028654  # sqrt(2/pi)
    k1 = 0.044715
    x3 = acc * acc * acc
    inner = k0 * (acc + k1 * x3)
    # tanh via exp
    e2 = tl.exp(2.0 * inner)
    tanh_val = (e2 - 1.0) / (e2 + 1.0)
    acc = 0.5 * acc * (1.0 + tanh_val)

    # store output in NCHW layout: out[n, oc, oh, ow]
    out_base = pid_n * C_out * H_out * W_out
    out_offsets = out_base + oc_offs[None, :] * (H_out * W_out) + oh[:, None] * W_out + ow[:, None]
    mask = hw_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self._cached_version = None
        self._w_nhwc_cuda = None
        self._b_cuda = None
        self._m_cuda = None

    def _refresh_cache(self):
        w = self.conv.weight.detach()
        b = self.conv.bias.detach()
        m = self.multiplier.detach()
        self._w_nhwc_cuda = w.permute(0, 2, 3, 1).contiguous().cuda()
        self._b_cuda = b.contiguous().cuda()
        self._m_cuda = m.contiguous().view(-1).cuda()
        self._cached_version = (
            self.conv.weight._version,
            self.conv.bias._version,
            self.multiplier._version,
        )

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        x = x.contiguous()
        N, C_in, H, W = x.shape
        C_out = self.out_channels
        KH = KW = self.kernel_size
        H_out = H - KH + 1
        W_out = W - KW + 1

        cur_version = (
            self.conv.weight._version,
            self.conv.bias._version,
            self.multiplier._version,
        )
        if self._cached_version != cur_version:
            self._refresh_cache()

        out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

        grid = lambda META: (
            N,
            triton.cdiv(C_out, META['BLOCK_OC']),
            triton.cdiv(H_out * W_out, META['BLOCK_HW']),
        )

        conv2d_fused_nhwc_kernel[grid](
            x, self._w_nhwc_cuda, self._b_cuda, self._m_cuda, out,
            N, C_in, H, W,
            C_out, H_out, W_out,
            KH=KH, KW=KW,
        )

        return out