import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=4, num_stages=2),
    ],
    key=['OH', 'OW', 'OC', 'IC', 'OD'],
)
@triton.jit
def conv3d_min_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, D, H, W,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # grid: (N * OC, ceil(OH*OW / BLOCK_HW))
    pid_no = tl.program_id(0)
    pid_hw = tl.program_id(1)

    n = pid_no // OC
    oc = pid_no % OC

    offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs < (OH * OW)
    oh = offs // OW
    ow = offs % OW

    # Precompute base x offset for this (n, oh, ow)
    # x layout: (N, IC, D, H, W), strides: (IC*D*H*W, D*H*W, H*W, W, 1)
    x_n_base = n * (IC * D * H * W)
    x_hw_base = oh * W + ow  # (BLOCK_HW,)

    bias = tl.load(b_ptr + oc)
    min_val = tl.full((BLOCK_HW,), float('inf'), dtype=tl.float32)

    # Loop over output depth (reduce along OD with min)
    for od in range(0, OD):
        acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)
        for ic in tl.static_range(0, IC):
            for kd in tl.static_range(0, KD):
                for kh in tl.static_range(0, KH):
                    for kw in tl.static_range(0, KW):
                        x_idx = x_n_base + ic * (D * H * W) + (od + kd) * (H * W) + kh * W + kw + x_hw_base
                        w_idx = ((oc * IC + ic) * KD + kd) * KH * KW + kh * KW + kw
                        x_val = tl.load(x_ptr + x_idx, mask=mask_hw, other=0.0)
                        w_val = tl.load(w_ptr + w_idx)
                        acc += x_val * w_val
        acc = acc + bias
        min_val = tl.minimum(min_val, acc)

    # Store result
    out_idx = ((n * OC + oc) * OH + oh) * OW + ow
    tl.store(out_ptr + out_idx, min_val, mask=mask_hw)


@triton.jit
def softmax_kernel(
    x_ptr, out_ptr,
    N, C, S,
    BLOCK_C: tl.constexpr,
):
    # grid: (N, S)
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    base = pid_n * C * S + pid_s
    ptrs = x_ptr + base + offs_c * S

    x = tl.load(ptrs, mask=mask_c, other=-float('inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask_c, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    out_ptrs = out_ptr + base + offs_c * S
    tl.store(out_ptrs, y, mask=mask_c)


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

        N, IC, D, H, W = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        OD = D - KD + 1
        OH = H - KH + 1
        OW = W - KW + 1

        # After conv: (N, OC, OD, OH, OW)
        # min over dim=self.dim of conv output
        # self.dim=2 corresponds to OD dimension in conv output
        # Our fused kernel reduces over OD producing (N, OC, OH, OW)
        if self.dim != 2:
            # fallback
            y = self.conv(x)
            y = torch.min(y, dim=self.dim)[0]
            return torch.softmax(y, dim=1)

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.float32)

        grid = lambda META: (N * OC, triton.cdiv(OH * OW, META['BLOCK_HW']))
        conv3d_min_kernel[grid](
            x, w, b, out,
            N, IC, D, H, W,
            OC, OD, OH, OW,
            KD, KH, KW,
        )

        # Softmax along channel dim (dim=1)
        S = OH * OW
        out_flat = out.view(N, OC, S)
        sm = torch.empty_like(out_flat)

        BLOCK_C = triton.next_power_of_2(OC)
        grid2 = (N, S)
        softmax_kernel[grid2](
            out_flat, sm,
            N, OC, S,
            BLOCK_C=BLOCK_C,
            num_warps=1,
        )

        return sm.view(N, OC, OH, OW)