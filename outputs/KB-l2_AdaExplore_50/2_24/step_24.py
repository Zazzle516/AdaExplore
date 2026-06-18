import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_min_softmax_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC,
    D, H, W,
    OC, OD, OH, OW,
    K: tl.constexpr,           # IC * KD * KH * KW
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # grid: (N, ceil(OH*OW/BLOCK_HW))
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < (OH * OW)
    oh = offs_hw // OW
    ow = offs_hw % OW

    offs_oc = tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    # weight is laid out as [OC, K] (row-major)
    # We will gather it once per (kd,kh,kw,ic) row -> shape [BLOCK_OC]
    # x is [N, IC, D, H, W]
    # for each od, accumulate [BLOCK_HW, BLOCK_OC] = sum over k of x[:, k] * w[:, k]^T

    # Load bias [BLOCK_OC]
    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)

    # Running min: [BLOCK_HW, BLOCK_OC]
    min_val = tl.full((BLOCK_HW, BLOCK_OC), float('inf'), dtype=tl.float32)

    for od in range(0, OD):
        acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

        for ic in range(0, IC):
            for kd in range(0, KD):
                id_ = od + kd
                for kh in range(0, KH):
                    ih = oh + kh  # [BLOCK_HW]
                    for kw in range(0, KW):
                        iw = ow + kw  # [BLOCK_HW]
                        # x value [BLOCK_HW]
                        x_idx = ((pid_n * IC + ic) * D + id_) * H * W + ih * W + iw
                        x_val = tl.load(x_ptr + x_idx, mask=mask_hw, other=0.0)

                        # weight value [BLOCK_OC]
                        k_idx = ((ic * KD + kd) * KH + kh) * KW + kw
                        w_idx = offs_oc * K + k_idx
                        w_val = tl.load(w_ptr + w_idx, mask=mask_oc, other=0.0)

                        # outer add: [BLOCK_HW, 1] * [1, BLOCK_OC]
                        acc += x_val[:, None] * w_val[None, :]

        acc = acc + bias[None, :]
        # Mask out-of-range channels with +inf so they don't affect min
        acc = tl.where(mask_oc[None, :], acc, float('inf'))
        min_val = tl.minimum(min_val, acc)

    # Now min_val: [BLOCK_HW, BLOCK_OC], do softmax along channel axis
    # Mask invalid channels to -inf for the softmax max
    sm_in = tl.where(mask_oc[None, :], min_val, -float('inf'))
    m = tl.max(sm_in, axis=1)  # [BLOCK_HW]
    e = tl.exp(sm_in - m[:, None])
    e = tl.where(mask_oc[None, :], e, 0.0)
    s = tl.sum(e, axis=1)  # [BLOCK_HW]
    y = e / s[:, None]

    # Write output [N, OC, OH, OW]
    # out_idx[hw, oc] = ((n * OC + oc) * OH * OW) + hw_offset
    # store with mask hw & oc
    out_offset = pid_n * OC * OH * OW + offs_oc[None, :] * (OH * OW) + offs_hw[:, None]
    mask_store = mask_hw[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_offset, y, mask=mask_store)


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
        w = self.conv.weight.contiguous().cuda()  # [OC, IC, KD, KH, KW]
        b = self.conv.bias.contiguous().cuda()    # [OC]

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

        K = IC * KD * KH * KW
        # reshape weight to [OC, K]
        w_flat = w.view(OC, K).contiguous()

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.float32)

        BLOCK_HW = 64
        BLOCK_OC = triton.next_power_of_2(OC)

        grid = (N, triton.cdiv(OH * OW, BLOCK_HW))
        conv3d_min_softmax_kernel[grid](
            x, w_flat, b, out,
            N, IC,
            D, H, W,
            OC, OD, OH, OW,
            K,
            KD, KH, KW,
            BLOCK_HW=BLOCK_HW,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
            num_stages=2,
        )

        return out