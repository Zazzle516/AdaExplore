import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv3d_min_softmax_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, D, H, W,
    OC: tl.constexpr, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # grid: (N, ceil(OH*OW / BLOCK_HW))
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)

    offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs < (OH * OW)
    oh = offs // OW
    ow = offs % OW

    offs_c = tl.arange(0, OC)

    # Load all biases once: (OC,)
    bias = tl.load(b_ptr + offs_c)  # (OC,)

    # min accumulator: (BLOCK_HW, OC)
    min_val = tl.full((BLOCK_HW, OC), float('inf'), dtype=tl.float32)

    # Loop over output depth
    for od in range(0, OD):
        acc = tl.zeros((BLOCK_HW, OC), dtype=tl.float32)
        for ic in range(0, IC):
            for kd in range(0, KD):
                id_ = od + kd
                for kh in range(0, KH):
                    ih = oh + kh
                    for kw in range(0, KW):
                        iw = ow + kw
                        x_idx = ((pid_n * IC + ic) * D + id_) * H * W + ih * W + iw
                        x_val = tl.load(x_ptr + x_idx, mask=mask_hw, other=0.0)  # (BLOCK_HW,)
                        # weight indices for all OC: ((oc * IC + ic) * KD + kd) * KH * KW + kh * KW + kw
                        w_idx = (offs_c * IC + ic) * KD * KH * KW + kd * KH * KW + kh * KW + kw
                        w_val = tl.load(w_ptr + w_idx)  # (OC,)
                        acc += x_val[:, None] * w_val[None, :]
        acc = acc + bias[None, :]
        min_val = tl.minimum(min_val, acc)

    # Softmax across OC (axis=1)
    m = tl.max(min_val, axis=1)  # (BLOCK_HW,)
    e = tl.exp(min_val - m[:, None])
    s = tl.sum(e, axis=1)  # (BLOCK_HW,)
    y = e / s[:, None]

    # Store: output layout (N, OC, OH*OW), contiguous
    # out[n, oc, hw] = ...
    out_base = pid_n * OC * (OH * OW)
    out_ptrs = out_ptr + out_base + offs_c[None, :] * (OH * OW) + offs[:, None]
    store_mask = mask_hw[:, None] & (offs_c[None, :] < OC)
    tl.store(out_ptrs, y, mask=store_mask)


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

        if self.dim != 2:
            y = self.conv(x)
            y = torch.min(y, dim=self.dim)[0]
            return torch.softmax(y, dim=1)

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.float32)

        BLOCK_HW = 64
        grid = (N, triton.cdiv(OH * OW, BLOCK_HW))
        fused_conv3d_min_softmax_kernel[grid](
            x, w, b, out,
            N, IC, D, H, W,
            OC, OD, OH, OW,
            KD, KH, KW,
            BLOCK_HW=BLOCK_HW,
            num_warps=4,
            num_stages=2,
        )

        return out