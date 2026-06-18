import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=3),
    ],
    key=['N', 'C_in', 'D', 'H', 'W', 'C_out', 'OD', 'OH', 'OW'],
)
@triton.jit
def conv3d_min_softmax_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in: tl.constexpr, D, H, W,
    C_out: tl.constexpr, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    pid_hw = tl.program_id(1)

    hw_start = pid_hw * BLOCK_HW
    hw_offs = hw_start + tl.arange(0, BLOCK_HW)
    mask_hw = hw_offs < (OH * OW)

    oh = hw_offs // OW
    ow = hw_offs % OW

    c_offs = tl.arange(0, BLOCK_C)
    mask_c = c_offs < C_out

    bias = tl.load(b_ptr + c_offs, mask=mask_c, other=0.0)

    min_val = tl.full((BLOCK_HW, BLOCK_C), float('inf'), dtype=tl.float32)

    HW = H * W
    KHW = KH * KW
    KDHW = KD * KHW
    CKDHW = C_in * KDHW

    ohw_base = oh * W + ow

    w_oc_stride = c_offs * CKDHW

    # Pre-compute per-(ic,kd,kh,kw) offsets into x and w, only depending on od via id_=od+kd
    for od in range(0, OD):
        acc = tl.zeros((BLOCK_HW, BLOCK_C), dtype=tl.float32) + bias[None, :]
        for ic in tl.static_range(0, C_in):
            x_ic_base = (n * C_in + ic) * D * HW
            w_ic_base = w_oc_stride + ic * KDHW
            for kd in tl.static_range(0, KD):
                id_ = od + kd
                x_d_base = x_ic_base + id_ * HW
                w_kd_base = w_ic_base + kd * KHW
                for kh in tl.static_range(0, KH):
                    x_h_base = x_d_base + kh * W + ohw_base
                    w_kh_base = w_kd_base + kh * KW
                    for kw in tl.static_range(0, KW):
                        x_idx = x_h_base + kw
                        w_idx = w_kh_base + kw
                        x_val = tl.load(x_ptr + x_idx, mask=mask_hw, other=0.0)
                        w_val = tl.load(w_ptr + w_idx, mask=mask_c, other=0.0)
                        acc += x_val[:, None] * w_val[None, :]
        min_val = tl.minimum(min_val, acc)

    min_val = tl.where(mask_c[None, :], min_val, -float('inf'))

    m = tl.max(min_val, axis=1)
    e = tl.exp(min_val - m[:, None])
    e = tl.where(mask_c[None, :], e, 0.0)
    z = tl.sum(e, axis=1)
    y = e / z[:, None]

    out_base = n * C_out * (OH * OW)
    out_idx = out_base + c_offs[None, :] * (OH * OW) + hw_offs[:, None]
    store_mask = mask_hw[:, None] & mask_c[None, :]
    tl.store(out_ptr + out_idx, y, mask=store_mask)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, C_in, D, H, W = x.shape
        C_out = self.out_channels
        K = self.kernel_size
        OD = D - K + 1
        OH = H - K + 1
        OW = W - K + 1

        if self.dim != 2:
            y = self.conv(x)
            y = torch.min(y, dim=self.dim)[0]
            return torch.softmax(y, dim=1)

        out = torch.empty((N, C_out, OH, OW), device=x.device, dtype=torch.float32)

        BLOCK_C = _next_pow2(C_out)
        grid = lambda meta: (N, triton.cdiv(OH * OW, meta['BLOCK_HW']))
        conv3d_min_softmax_kernel[grid](
            x, w, b, out,
            N, C_in, D, H, W,
            C_out, OD, OH, OW,
            K, K, K,
            BLOCK_C=BLOCK_C,
        )
        return out