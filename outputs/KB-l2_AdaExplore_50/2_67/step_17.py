import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_gelu_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, OC: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
    OH: tl.constexpr, OW: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    INV_POOL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < (OH * OW)

    oh = sp_offs // OW
    ow = sp_offs % OW

    K = IC * KH * KW

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    for ic in tl.static_range(IC):
        for kh in tl.static_range(KH):
            for kw in tl.static_range(KW):
                ih = oh + kh
                iw = ow + kw
                x_off = pid_n * (IC * H * W) + ic * (H * W) + ih * W + iw
                x_val = tl.load(x_ptr + x_off, mask=sp_mask, other=0.0)
                k_idx = ic * (KH * KW) + kh * KW + kw
                w_col = tl.load(w_ptr + oc_offs * K + k_idx, mask=oc_mask, other=0.0)
                acc += w_col[:, None] * x_val[None, :]

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += bias[:, None]

    # GELU tanh approx
    k0 = 0.7978845608028654
    k1 = 0.044715
    x3 = acc * acc * acc
    inner = k0 * (acc + k1 * x3)
    e2 = tl.exp(2.0 * inner)
    tanh_v = (e2 - 1.0) / (e2 + 1.0)
    gelu = 0.5 * acc * (1.0 + tanh_v)

    gelu = tl.where(sp_mask[None, :], gelu, 0.0)
    partial = tl.sum(gelu, axis=1) * INV_POOL

    out_off = pid_n * OC + oc_offs
    tl.atomic_add(out_ptr + out_off, partial, mask=oc_mask)


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

        out = torch.zeros((N, OC), device=x.device, dtype=torch.float32)

        BLOCK_OC = 64
        BLOCK_SP = 256

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_SP))

        conv_gelu_pool_kernel[grid](
            x, w, b, out,
            N, IC, OC, H, W, OH, OW, KH, KW,
            BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
            INV_POOL=1.0 / float(OH * OW),
            num_warps=8, num_stages=2,
        )

        return out