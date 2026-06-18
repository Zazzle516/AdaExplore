import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['C_in', 'C_out', 'H_out', 'W_out', 'KH', 'KW'],
)
@triton.jit
def conv2d_fused_persistent_kernel(
    x_ptr, w_ptr, b_ptr, m_ptr, out_ptr,
    N, C_in, H, W,
    C_out, H_out, W_out,
    M_total,  # N * H_out * W_out
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_HW: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_oc = tl.program_id(0)
    pid_m = tl.program_id(1)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    m_offs = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)

    oc_mask = oc_offs < C_out
    m_mask = m_offs < M_total

    # decompose m_offs into (n, oh, ow)
    hw = H_out * W_out
    n_idx = m_offs // hw
    rem = m_offs % hw
    oh = rem // W_out
    ow = rem % W_out

    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    K = KH * KW * C_in
    k_offs = tl.arange(0, BLOCK_K)

    # NHWC layout: x[n, ih, iw, ic] = n*H*W*C_in + ih*W*C_in + iw*C_in + ic
    n_base = n_idx * (H * W * C_in)  # [BLOCK_HW]

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + k_offs  # [BLOCK_K]
        k_mask = k_idx < K

        ic = k_idx % C_in
        khw = k_idx // C_in
        kw = khw % KW
        kh = khw // KW

        ih = oh[:, None] + kh[None, :]
        iw = ow[:, None] + kw[None, :]
        x_offsets = n_base[:, None] + ih * (W * C_in) + iw * C_in + ic[None, :]
        x_mask = m_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

        w_offsets = oc_offs[None, :] * K + k_idx[:, None]
        w_mask = k_mask[:, None] & oc_mask[None, :]
        w_tile = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)

        acc += tl.dot(x_tile, w_tile)

    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    m_vals = tl.load(m_ptr + oc_offs, mask=oc_mask, other=0.0)

    acc = acc + b_vals[None, :]
    acc = acc * m_vals[None, :]

    # LeakyReLU(0.01)
    acc = tl.where(acc >= 0, acc, acc * 0.01)

    # GELU tanh approx
    k0 = 0.7978845608028654
    k1 = 0.044715
    x3 = acc * acc * acc
    inner = k0 * (acc + k1 * x3)
    e2 = tl.exp(2.0 * inner)
    tanh_val = (e2 - 1.0) / (e2 + 1.0)
    acc = 0.5 * acc * (1.0 + tanh_val)

    # store to NCHW out: out[n, oc, oh, ow] = n*C_out*H_out*W_out + oc*H_out*W_out + oh*W_out + ow
    out_offsets = (n_idx[:, None] * (C_out * H_out * W_out)
                   + oc_offs[None, :] * (H_out * W_out)
                   + oh[:, None] * W_out
                   + ow[:, None])
    mask = m_mask[:, None] & oc_mask[None, :]
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

    def forward(self, x):
        x = x.contiguous().cuda()
        N, C_in, H, W = x.shape
        C_out = self.out_channels
        KH = KW = self.kernel_size
        H_out = H - KH + 1
        W_out = W - KW + 1

        # NCHW -> NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        w = self.conv.weight  # (OC, C_in, KH, KW)
        w_nhwc = w.permute(0, 2, 3, 1).contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()
        m = self.multiplier.contiguous().view(-1).cuda()

        out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

        M_total = N * H_out * W_out

        grid = lambda META: (
            triton.cdiv(C_out, META['BLOCK_OC']),
            triton.cdiv(M_total, META['BLOCK_HW']),
        )

        conv2d_fused_persistent_kernel[grid](
            x_nhwc, w_nhwc, b, m, out,
            N, C_in, H, W,
            C_out, H_out, W_out,
            M_total,
            KH=KH, KW=KW,
        )

        return out