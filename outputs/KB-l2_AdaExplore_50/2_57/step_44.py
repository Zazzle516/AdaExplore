import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_relu_hardswish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oh = sp_offs // OW
    ow = sp_offs % OW

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (OH * OW)

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # iterate kh, kw, ic
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh  # padding=0
            iw = ow + kw
            for ic in tl.static_range(0, IC_C):
                # load input [BLOCK_SP]
                x_off = pid_n * (IC * H * W) + ic * (H * W) + ih * W + iw
                x_val = tl.load(x_ptr + x_off, mask=sp_mask, other=0.0)
                # load weight [BLOCK_OC]
                w_off = oc_offs * (IC_C * KH * KW) + ic * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                acc += x_val[:, None] * w_val[None, :]

    # add bias
    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b_val[None, :]

    # relu
    acc = tl.maximum(acc, 0.0)
    # hardswish: x * clamp((x+3)/6, 0, 1)
    hs = tl.minimum(tl.maximum((acc + 3.0) / 6.0, 0.0), 1.0)
    out = acc * hs

    # store
    out_off = (pid_n * OC * OH * OW
               + oc_offs[None, :] * (OH * OW)
               + sp_offs[:, None])
    mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()
        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_SP = 128

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_SP))

        conv_relu_hardswish_kernel[grid](
            x, w, b, out,
            N, IC, H, W,
            OC, OH, OW,
            KH, KW,
            IC,
            BLOCK_OC, BLOCK_SP,
            num_warps=4, num_stages=2,
        )
        return out