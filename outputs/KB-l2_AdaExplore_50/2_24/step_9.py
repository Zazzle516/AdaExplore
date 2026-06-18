import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_min_softmax_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in,
    D: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C_out: tl.constexpr, OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_HW: tl.constexpr, C_PAD: tl.constexpr,
):
    # grid: (N, num_hw_tiles)
    n = tl.program_id(0)
    pid_hw = tl.program_id(1)

    hw_start = pid_hw * BLOCK_HW
    hw_offs = hw_start + tl.arange(0, BLOCK_HW)
    mask_hw = hw_offs < (OH * OW)

    oh = hw_offs // OW
    ow = hw_offs % OW

    c_offs = tl.arange(0, C_PAD)
    mask_c = c_offs < C_out

    # load bias [C_PAD]
    bias = tl.load(b_ptr + c_offs, mask=mask_c, other=0.0)

    # min_val [C_PAD, BLOCK_HW]
    min_val = tl.full((C_PAD, BLOCK_HW), float('inf'), dtype=tl.float32)

    KVOL: tl.constexpr = KD * KH * KW
    K_total = C_in * KVOL

    for od in range(0, OD):
        acc = bias[:, None] + tl.zeros((C_PAD, BLOCK_HW), dtype=tl.float32)
        for k in range(0, K_total):
            ic = k // KVOL
            krem = k % KVOL
            kd = krem // (KH * KW)
            khw = krem % (KH * KW)
            kh = khw // KW
            kw = khw % KW

            id_ = od + kd
            ih = oh + kh
            iw = ow + kw

            x_idx = ((n * C_in + ic) * D + id_) * H * W + ih * W + iw
            x_val = tl.load(x_ptr + x_idx, mask=mask_hw, other=0.0)  # [BLOCK_HW]

            # weight: [C_out] for this (ic, kd, kh, kw)
            w_idx = ((c_offs * C_in + ic) * KD + kd) * KH * KW + kh * KW + kw
            w_val = tl.load(w_ptr + w_idx, mask=mask_c, other=0.0)  # [C_PAD]

            acc += w_val[:, None] * x_val[None, :]

        min_val = tl.minimum(min_val, acc)

    # softmax along C axis
    min_val = tl.where(mask_c[:, None], min_val, float('inf'))
    m = tl.min(min_val, axis=0)  # [BLOCK_HW]
    shifted = min_val - m[None, :]
    shifted = tl.where(mask_c[:, None], shifted, -float('inf'))
    e = tl.exp(shifted)
    z = tl.sum(e, axis=0)
    y = e / z[None, :]

    # store
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

        # pad C_out to next pow2
        C_PAD = 1
        while C_PAD < C_out:
            C_PAD *= 2

        BLOCK_HW = 128
        grid = (N, triton.cdiv(OH * OW, BLOCK_HW))
        conv3d_min_softmax_kernel[grid](
            x, w, b, out,
            N, C_in,
            D, H, W,
            C_out, OD, OH, OW,
            K, K, K,
            BLOCK_HW=BLOCK_HW, C_PAD=C_PAD,
            num_warps=4, num_stages=2,
        )
        return out