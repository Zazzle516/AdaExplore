import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_min_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, D, H, W,
    C_out, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # grid: (N * C_out, OH * OW blocks)
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    n = pid_nc // C_out
    oc = pid_nc % C_out

    hw_start = pid_hw * BLOCK_HW
    hw_offs = hw_start + tl.arange(0, BLOCK_HW)
    mask_hw = hw_offs < (OH * OW)

    oh = hw_offs // OW
    ow = hw_offs % OW

    # load bias
    bias = tl.load(b_ptr + oc)

    # We need min over OD of conv result. Initialize to large.
    min_val = tl.full((BLOCK_HW,), float('inf'), dtype=tl.float32)

    for od in range(0, OD):
        acc = tl.zeros((BLOCK_HW,), dtype=tl.float32) + bias
        for ic in range(0, C_in):
            for kd in range(0, KD):
                for kh in range(0, KH):
                    for kw in range(0, KW):
                        id_ = od + kd
                        ih = oh + kh
                        iw = ow + kw
                        x_idx = ((n * C_in + ic) * D + id_) * H * W + ih * W + iw
                        w_idx = ((oc * C_in + ic) * KD + kd) * KH * KW + kh * KW + kw
                        x_val = tl.load(x_ptr + x_idx, mask=mask_hw, other=0.0)
                        w_val = tl.load(w_ptr + w_idx)
                        acc += x_val * w_val
        min_val = tl.minimum(min_val, acc)

    # store: output shape (N, C_out, OH, OW)
    out_idx = ((n * C_out + oc) * OH * OW) + hw_offs
    tl.store(out_ptr + out_idx, min_val, mask=mask_hw)


@triton.jit
def softmax_kernel(
    x_ptr, out_ptr,
    N, C, S,
    BLOCK_C: tl.constexpr,
):
    # grid: (N, S)
    n = tl.program_id(0)
    s = tl.program_id(1)

    c_offs = tl.arange(0, BLOCK_C)
    mask = c_offs < C

    base = n * C * S + s
    x = tl.load(x_ptr + base + c_offs * S, mask=mask, other=-float('inf'))

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    z = tl.sum(e, axis=0)
    y = e / z
    tl.store(out_ptr + base + c_offs * S, y, mask=mask)


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
            # fallback
            y = self.conv(x)
            y = torch.min(y, dim=self.dim)[0]
            return torch.softmax(y, dim=1)

        out = torch.empty((N, C_out, OH, OW), device=x.device, dtype=torch.float32)

        BLOCK_HW = 64
        grid = (N * C_out, triton.cdiv(OH * OW, BLOCK_HW))
        conv3d_min_kernel[grid](
            x, w, b, out,
            N, C_in, D, H, W,
            C_out, OD, OH, OW,
            K, K, K,
            BLOCK_HW=BLOCK_HW,
            num_warps=2,
        )

        # softmax along channel dim
        S = OH * OW
        out_sm = torch.empty_like(out)
        BLOCK_C = _next_pow2(C_out)
        softmax_kernel[(N, S)](
            out, out_sm,
            N, C_out, S,
            BLOCK_C=BLOCK_C,
            num_warps=1,
        )
        return out_sm