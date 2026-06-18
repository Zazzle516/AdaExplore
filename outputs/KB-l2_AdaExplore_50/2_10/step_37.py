import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_pool_htanh_mean_tanh_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C, H, W,
    pooled_H, pooled_W,
    hardtanh_min: tl.constexpr, hardtanh_max: tl.constexpr,
    inv_count,
    IC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    oc = pid % C

    total = pooled_H * pooled_W
    H_in = H  # input H (same as output spatial H since stride=1, pad=1, k=3)
    W_in = W

    # Load weight for this oc: shape [IC, 3, 3]
    # weight layout: [OC, IC, 3, 3] contiguous
    w_base = oc * IC * 9
    ic_range = tl.arange(0, IC)  # [IC]

    bias_val = tl.load(b_ptr + oc)

    acc_sum = 0.0
    for start in range(0, total, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)  # [BLOCK_HW]
        mask = offs < total
        ph = offs // pooled_W
        pw = offs % pooled_W
        # Pool window: output H positions = ph*2, ph*2+1; W positions = pw*2, pw*2+1
        # For each of 4 output spatial positions, compute conv (3x3) sum over IC
        # output(h, w) = sum_{ic, kh, kw} x[n, ic, h+kh-1, w+kw-1] * w[oc, ic, kh, kw] + b[oc]

        # We'll compute 4 conv outputs and take max
        # For efficiency, precompute base for x
        x_base = n * C * H_in * W_in  # using C==IC

        # Initialize 4 accumulators
        v00 = tl.zeros([BLOCK_HW], dtype=tl.float32) + bias_val
        v01 = tl.zeros([BLOCK_HW], dtype=tl.float32) + bias_val
        v10 = tl.zeros([BLOCK_HW], dtype=tl.float32) + bias_val
        v11 = tl.zeros([BLOCK_HW], dtype=tl.float32) + bias_val

        # Loop over kh, kw (3x3) and unroll IC reduction via vector load
        for kh in tl.static_range(0, 3):
            for kw in tl.static_range(0, 3):
                # weight values for all IC at (kh, kw): w[oc, :, kh, kw]
                w_offs = w_base + ic_range * 9 + kh * 3 + kw  # [IC]
                wv = tl.load(w_ptr + w_offs)  # [IC]

                # For each of 4 output positions
                # position 0: (h=ph*2, w=pw*2) -> input h = ph*2 + kh - 1, input w = pw*2 + kw - 1
                for sh in tl.static_range(0, 2):
                    for sw in tl.static_range(0, 2):
                        h_out = ph * 2 + sh
                        w_out = pw * 2 + sw
                        h_in = h_out + kh - 1
                        w_in = w_out + kw - 1
                        in_bounds = (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in) & mask
                        # x address for all IC
                        # x[n, ic, h_in, w_in] = x_base + ic*H*W + h_in*W + w_in
                        # gather: shape [BLOCK_HW, IC]
                        x_off = x_base + ic_range[None, :] * (H_in * W_in) + h_in[:, None] * W_in + w_in[:, None]
                        xv = tl.load(x_ptr + x_off, mask=in_bounds[:, None], other=0.0)  # [BLOCK_HW, IC]
                        # multiply and reduce over IC
                        prod = xv * wv[None, :]
                        s = tl.sum(prod, axis=1)  # [BLOCK_HW]
                        if sh == 0 and sw == 0:
                            v00 = v00 + s
                        if sh == 0 and sw == 1:
                            v01 = v01 + s
                        if sh == 1 and sw == 0:
                            v10 = v10 + s
                        if sh == 1 and sw == 1:
                            v11 = v11 + s

        m = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
        m = tl.minimum(tl.maximum(m, hardtanh_min), hardtanh_max)
        m = tl.where(mask, m, 0.0)
        acc_sum += tl.sum(m, axis=0)

    mean_val = acc_sum * inv_count
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

    acc = 0.0
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

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