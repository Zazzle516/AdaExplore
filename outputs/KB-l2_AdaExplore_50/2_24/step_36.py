import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_min_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, D, H, W,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # grid: (N, OC, ceil(OH*OW / BLOCK_HW))
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs < (OH * OW)
    oh = offs // OW
    ow = offs % OW

    bias = tl.load(b_ptr + pid_oc)

    # accumulator for min over depth
    INF = float('inf')
    min_val = tl.full((BLOCK_HW,), INF, dtype=tl.float32)

    # loop over output depth
    for od in range(0, OD):
        acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)
        # loop over input channels and kernel
        for ic in range(0, IC):
            for kd in range(0, KD):
                id_ = od + kd
                for kh in range(0, KH):
                    ih = oh + kh
                    for kw in range(0, KW):
                        iw = ow + kw
                        x_off = (((pid_n * IC + ic) * D + id_) * H + ih) * W + iw
                        w_off = (((pid_oc * IC + ic) * KD + kd) * KH + kh) * KW + kw
                        xv = tl.load(x_ptr + x_off, mask=mask, other=0.0)
                        wv = tl.load(w_ptr + w_off)
                        acc += xv * wv
        acc = acc + bias
        min_val = tl.minimum(min_val, acc)

    out_off = (pid_n * OC + pid_oc) * (OH * OW) + offs
    tl.store(out_ptr + out_off, min_val, mask=mask)


@triton.jit
def softmax_kernel(
    inp_ptr, out_ptr,
    N, C, S,
    BLOCK_C: tl.constexpr,
):
    # grid: (N, S)
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    c_offs = tl.arange(0, BLOCK_C)
    mask = c_offs < C
    base = pid_n * C * S + pid_s

    vals = tl.load(inp_ptr + base + c_offs * S, mask=mask, other=-float('inf'))
    m = tl.max(vals, axis=0)
    e = tl.exp(vals - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(out_ptr + base + c_offs * S, out, mask=mask)


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

            BLOCK_HW = 64
            grid = (N, OC, (OH * OW + BLOCK_HW - 1) // BLOCK_HW)
            conv3d_min_kernel[grid](
                x, w, b, out,
                N, IC, D, H, W,
                OC, OD, OH, OW,
                KD, KH, KW,
                BLOCK_HW=BLOCK_HW,
                num_warps=4,
            )

            # softmax along channel dim
            S = OH * OW
            out_flat = out.view(N, OC, S)
            sm_out = torch.empty_like(out_flat)
            BLOCK_C = _next_pow2(OC)
            grid_sm = (N, S)
            softmax_kernel[grid_sm](
                out_flat, sm_out,
                N, OC, S,
                BLOCK_C=BLOCK_C,
                num_warps=2,
            )
            return sm_out.view(N, OC, OH, OW)
        else:
            # fallback
            y = self.conv(x)
            y = torch.min(y, dim=self.dim)[0]
            y = torch.softmax(y, dim=1)
            return y