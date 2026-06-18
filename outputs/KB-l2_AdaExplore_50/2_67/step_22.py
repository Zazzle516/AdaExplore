import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_gelu_avgpool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, H, W, OH, OW,
    inv_npix,
    BLOCK_PIX: tl.constexpr,
    IC: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    OC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)

    oc = pid_oc  # one program per output channel

    # Preload weights for this OC: shape [IC, KH, KW] => IC*KH*KW floats
    w_offs = oc * (IC * KH * KW) + tl.arange(0, IC * KH * KW)
    w_vals = tl.load(w_ptr + w_offs)  # [IC*KH*KW]

    bias = tl.load(b_ptr + oc)

    npix = OH * OW
    n_blocks = (npix + BLOCK_PIX - 1) // BLOCK_PIX

    acc_sum = tl.zeros((1,), dtype=tl.float32)
    total = 0.0

    for blk in range(0, n_blocks):
        pix_offs = blk * BLOCK_PIX + tl.arange(0, BLOCK_PIX)
        pix_mask = pix_offs < npix
        oh = pix_offs // OW
        ow = pix_offs % OW

        acc = tl.zeros((BLOCK_PIX,), dtype=tl.float32)

        for ic in tl.static_range(0, IC):
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    ih = oh + kh
                    iw = ow + kw
                    x_off = pid_n * (IC * H * W) + ic * (H * W) + ih * W + iw
                    x_vals = tl.load(x_ptr + x_off, mask=pix_mask, other=0.0)
                    w_idx = ic * (KH * KW) + kh * KW + kw
                    w_scalar = tl.load(w_ptr + oc * (IC * KH * KW) + w_idx)
                    acc += x_vals * w_scalar

        acc = acc + bias
        inv_sqrt2 = 0.70710678118654752440
        gelu_out = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))
        gelu_out = tl.where(pix_mask, gelu_out, 0.0)
        total += tl.sum(gelu_out, axis=0)

    result = total * inv_npix
    tl.store(out_ptr + pid_n * OC + oc, result)


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
        npix = OH * OW
        inv_npix = 1.0 / npix

        out = torch.empty((N, OC), device=x.device, dtype=torch.float32)

        BLOCK_PIX = 1024

        grid = (N, OC)

        conv_gelu_avgpool_kernel[grid](
            x, w, b, out,
            N, H, W, OH, OW,
            inv_npix,
            BLOCK_PIX=BLOCK_PIX,
            IC=IC,
            KH=KH,
            KW=KW,
            OC=OC,
            num_warps=8,
            num_stages=3,
        )

        return out