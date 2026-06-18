import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 32}, num_warps=2, num_stages=3),
    ],
    key=['OH', 'OW', 'C_out', 'C_in'],
)
@triton.jit
def conv3d_min_softmax_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in: tl.constexpr,
    D: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C_out: tl.constexpr, OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    C_PAD: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    pid_hw = tl.program_id(1)

    hw_start = pid_hw * BLOCK_HW
    hw_offs = hw_start + tl.arange(0, BLOCK_HW)
    mask_hw = hw_offs < (OH * OW)

    oh = hw_offs // OW
    ow = hw_offs % OW

    c_offs = tl.arange(0, C_PAD)
    mask_c = c_offs < C_out

    bias = tl.load(b_ptr + c_offs, mask=mask_c, other=0.0)

    min_val = tl.full((C_PAD, BLOCK_HW), float('inf'), dtype=tl.float32)

    HW = H * W
    KHW = KH * KW
    KDHW = KD * KHW
    CKDHW = C_in * KDHW

    # spatial base per output element (within an input depth plane)
    ohw_base = oh * W + ow  # (BLOCK_HW,)

    # weight base per (c_out): c_offs * C_in * KD * KH * KW
    w_c_base = c_offs * CKDHW  # (C_PAD,)

    n_base = n * C_in * D * HW

    for od in range(0, OD):
        acc = tl.zeros((C_PAD, BLOCK_HW), dtype=tl.float32) + bias[:, None]
        for ic in tl.static_range(0, C_in):
            x_ic_base = n_base + ic * D * HW
            w_ic_base = w_c_base + ic * KDHW  # (C_PAD,)
            for kd in tl.static_range(0, KD):
                id_ = od + kd
                x_d_base = x_ic_base + id_ * HW
                w_kd_base = w_ic_base + kd * KHW  # (C_PAD,)
                for kh in tl.static_range(0, KH):
                    x_h_base = x_d_base + kh * W + ohw_base  # (BLOCK_HW,)
                    w_kh_base = w_kd_base + kh * KW  # (C_PAD,)
                    for kw in tl.static_range(0, KW):
                        x_val = tl.load(x_ptr + x_h_base + kw, mask=mask_hw, other=0.0)
                        w_val = tl.load(w_ptr + w_kh_base + kw, mask=mask_c, other=0.0)
                        acc += w_val[:, None] * x_val[None, :]
        min_val = tl.minimum(min_val, acc)

    # softmax along C axis
    min_val = tl.where(mask_c[:, None], min_val, -float('inf'))
    m = tl.max(min_val, axis=0)  # (BLOCK_HW,)
    shifted = min_val - m[None, :]
    shifted = tl.where(mask_c[:, None], shifted, -float('inf'))
    e = tl.exp(shifted)
    z = tl.sum(e, axis=0)
    y = e / z[None, :]

    out_base = (n * C_out) * (OH * OW)
    out_idx = out_base + c_offs[:, None] * (OH * OW) + hw_offs[None, :]
    store_mask = mask_c[:, None] & mask_hw[None, :]
    tl.store(out_ptr + out_idx, y, mask=store_mask)


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

        C_PAD = 1
        while C_PAD < C_out:
            C_PAD *= 2

        grid = lambda META: (N, triton.cdiv(OH * OW, META['BLOCK_HW']))
        conv3d_min_softmax_kernel[grid](
            x, w, b, out,
            N, C_in,
            D, H, W,
            C_out, OD, OH, OW,
            K, K, K,
            C_PAD=C_PAD,
        )
        return out