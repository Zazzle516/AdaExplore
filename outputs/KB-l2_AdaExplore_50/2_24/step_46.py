import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_HW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=2),
    ],
    key=['OH', 'OW', 'OC', 'IC', 'OD'],
)
@triton.jit
def fused_conv_min_softmax_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, D, H, W,
    OC: tl.constexpr, OD: tl.constexpr, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)

    offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    S = OH * OW
    mask = offs < S
    oh = offs // OW
    ow = offs % OW

    oc_range = tl.arange(0, OC)
    od_range = tl.arange(0, OD)
    HW = H * W

    INF = float('inf')
    mins = tl.full([BLOCK_HW, OC], INF, dtype=tl.float32)

    x_n_base = pid_n * IC * D * HW

    for oc in range(OC):
        bias = tl.load(b_ptr + oc)
        acc_od = tl.zeros([BLOCK_HW, OD], dtype=tl.float32)
        for ic in range(IC):
            x_ic_base = x_n_base + ic * D * HW
            w_ic_base = (oc * IC + ic) * KD * KH * KW
            for kd in range(KD):
                d_off = (od_range + kd) * HW  # [OD]
                for kh in range(KH):
                    ih = oh + kh
                    for kw in range(KW):
                        iw = ow + kw
                        w_val = tl.load(w_ptr + w_ic_base + (kd * KH + kh) * KW + kw)
                        sp_off = ih * W + iw  # [BLOCK_HW]
                        x_off = x_ic_base + sp_off[:, None] + d_off[None, :]
                        x_vals = tl.load(x_ptr + x_off, mask=mask[:, None], other=0.0)
                        acc_od += x_vals * w_val
        acc_od = acc_od + bias
        min_oc = tl.min(acc_od, axis=1)  # [BLOCK_HW]
        oc_mask = (oc_range == oc)[None, :]
        mins = tl.where(oc_mask, min_oc[:, None], mins)

    # softmax across OC dim
    m = tl.max(mins, axis=1)
    e = tl.exp(mins - m[:, None])
    s_sum = tl.sum(e, axis=1)
    out_vals = e / s_sum[:, None]

    out_off = (pid_n * OC + oc_range)[None, :] * S + offs[:, None]
    tl.store(out_ptr + out_off, out_vals, mask=mask[:, None])


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

        N, IC, D, H, W = x.shape
        OC, _, KD, KH, KW = w.shape

        if self.dim == 2:
            OD = D - KD + 1
            OH = H - KH + 1
            OW = W - KW + 1

            out = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.float32)

            S = OH * OW
            grid = lambda META: (N, (S + META['BLOCK_HW'] - 1) // META['BLOCK_HW'])
            fused_conv_min_softmax_kernel[grid](
                x, w, b, out,
                N, IC, D, H, W,
                OC, OD, OH, OW,
                KD, KH, KW,
            )
            return out
        else:
            # fallback
            y = self.conv(x)
            y = torch.min(y, dim=self.dim)[0]
            y = torch.softmax(y, dim=1)
            return y