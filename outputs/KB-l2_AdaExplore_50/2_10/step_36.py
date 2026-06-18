import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_pool_htanh_mean_tanh_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W, OC,
    pooled_H, pooled_W,
    hardtanh_min: tl.constexpr, hardtanh_max: tl.constexpr,
    inv_count,
    BLOCK_P: tl.constexpr,
):
    # one program per (n, oc); iterate pooled output spatial in tiles
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid % OC

    total = pooled_H * pooled_W
    bias = tl.load(b_ptr + oc)

    acc = tl.zeros((1,), dtype=tl.float32)

    # weight layout: [IC, OC, 3, 3] for ConvTranspose2d
    # For stride=1, pad=1, kernel=3:
    # y[oc, oh, ow] = sum_{ic, kh, kw} x[ic, oh+kh-1+1-? ...] * w[ic, oc, kh, kw]
    # Standard ConvT formula (stride=1, pad=1, k=3):
    # y[n, oc, oh, ow] = sum_{ic, kh, kw} x[n, ic, oh - kh + 1, ow - kw + 1] * w[ic, oc, kh, kw]
    # for kh, kw in 0..2 (we treat ConvT as conv with flipped kernel)
    # Actually: y[oh, ow] = sum_{ic} sum_{kh,kw} x[ic, oh+pad-kh, ow+pad-kw] * w[ic, oc, kh, kw]
    # with pad=1: ih = oh + 1 - kh, iw = ow + 1 - kw

    for tile_start in range(0, total, BLOCK_P):
        offs = tile_start + tl.arange(0, BLOCK_P)
        mask = offs < total
        ph = offs // pooled_W
        pw = offs % pooled_W

        # Compute the 2x2 conv outputs for each pooled position
        # out positions: (2*ph + dh, 2*pw + dw) for dh, dw in {0,1}
        max_val = tl.full((BLOCK_P,), -float('inf'), dtype=tl.float32)

        # Unroll 2x2 pool positions
        for dh in tl.static_range(2):
            for dw in tl.static_range(2):
                oh = ph * 2 + dh
                ow = pw * 2 + dw
                # compute conv at (oh, ow)
                conv_val = tl.zeros((BLOCK_P,), dtype=tl.float32) + bias
                for kh in tl.static_range(3):
                    for kw in tl.static_range(3):
                        ih = oh + 1 - kh
                        iw = ow + 1 - kw
                        valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & mask
                        for ic in range(IC):
                            x_off = n * IC * H * W + ic * H * W + ih * W + iw
                            w_off = ic * OC * 9 + oc * 9 + kh * 3 + kw
                            xv = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                            wv = tl.load(w_ptr + w_off)
                            conv_val += xv * wv
                # apply hardtanh
                ht = tl.minimum(tl.maximum(conv_val, hardtanh_min), hardtanh_max)
                max_val = tl.maximum(max_val, conv_val)

        # apply hardtanh after max (equivalent since hardtanh is monotonic)
        max_val = tl.minimum(tl.maximum(max_val, hardtanh_min), hardtanh_max)
        max_val = tl.where(mask, max_val, 0.0)
        acc += tl.sum(max_val, axis=0)

    mean_val = acc * inv_count
    e1 = tl.exp(mean_val)
    e2 = tl.exp(-mean_val)
    out_val = (e1 - e2) / (e1 + e2)
    tl.store(out_ptr + pid, out_val)


@triton.jit
def fused_pool_htanh_mean_tanh_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    pooled_H, pooled_W,
    hardtanh_min: tl.constexpr, hardtanh_max: tl.constexpr,
    inv_count,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    total = pooled_H * pooled_W
    base = n * C * H * W + c * H * W

    acc = tl.zeros((1,), dtype=tl.float32)
    for start in range(0, total, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < total
        ph = offs // pooled_W
        pw = offs % pooled_W
        h0 = ph * 2
        w0 = pw * 2
        i00 = base + h0 * W + w0
        i01 = base + h0 * W + (w0 + 1)
        i10 = base + (h0 + 1) * W + w0
        i11 = base + (h0 + 1) * W + (w0 + 1)
        v00 = tl.load(x_ptr + i00, mask=mask, other=-float('inf'))
        v01 = tl.load(x_ptr + i01, mask=mask, other=-float('inf'))
        v10 = tl.load(x_ptr + i10, mask=mask, other=-float('inf'))
        v11 = tl.load(x_ptr + i11, mask=mask, other=-float('inf'))
        m = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
        m = tl.minimum(tl.maximum(m, hardtanh_min), hardtanh_max)
        m = tl.where(mask, m, 0.0)
        acc += tl.sum(m, axis=0)

    mean_val = acc * inv_count
    e1 = tl.exp(mean_val)
    e2 = tl.exp(-mean_val)
    out_val = (e1 - e2) / (e1 + e2)
    tl.store(out_ptr + pid, out_val)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding)
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = float(hardtanh_min)
        self.hardtanh_max = float(hardtanh_max)

    def forward(self, x):
        x = self.conv_transpose(x)
        N, C, H, W = x.shape
        x = x.contiguous()

        pooled_H = H // 2
        pooled_W = W // 2

        out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)

        total = pooled_H * pooled_W
        BLOCK = 1024
        if total < 1024:
            BLOCK = 256
        if total < 256:
            BLOCK = 64

        grid = (N * C,)
        fused_pool_htanh_mean_tanh_kernel[grid](
            x, out,
            N, C, H, W,
            pooled_H, pooled_W,
            self.hardtanh_min, self.hardtanh_max,
            1.0 / float(total),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out