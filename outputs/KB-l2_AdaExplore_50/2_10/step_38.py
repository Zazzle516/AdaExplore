import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_pool_htanh_mean_tanh_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, C_out, H, W,
    pooled_H, pooled_W,
    hardtanh_min, hardtanh_max,
    inv_count,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # one program per (n, c_out)
    pid = tl.program_id(0)
    n = pid // C_out
    oc = pid % C_out

    total = pooled_H * pooled_W
    out_HW = H * W
    in_HW = H * W
    # input strides: n*C_in*H*W + ci*H*W
    in_base_n = n * C_in * in_HW
    # weight layout: ConvTranspose2d weight is (C_in, C_out, 3, 3)
    # Equivalent regular conv with stride=1, padding=1 uses weight[ci, oc, 2-kh, 2-kw]
    # We'll precompute offsets for the 9 flipped weight positions per (ci, oc).
    # weight stride: ci*(C_out*9) + oc*9 + kh*3 + kw

    bias_val = tl.load(b_ptr + oc)

    acc_sum = 0.0

    for p_start in range(0, total, BLOCK_P):
        offs = p_start + tl.arange(0, BLOCK_P)
        mask_p = offs < total
        ph = offs // pooled_W
        pw = offs % pooled_W
        # 2x2 pool: pre-pool positions are (2*ph + dh, 2*pw + dw) for dh,dw in {0,1}
        h0 = ph * 2
        w0 = pw * 2

        # accumulators for the four pre-pool conv outputs
        conv00 = tl.zeros([BLOCK_P], dtype=tl.float32) + bias_val
        conv01 = tl.zeros([BLOCK_P], dtype=tl.float32) + bias_val
        conv10 = tl.zeros([BLOCK_P], dtype=tl.float32) + bias_val
        conv11 = tl.zeros([BLOCK_P], dtype=tl.float32) + bias_val

        # Loop over input channels
        for ci_start in range(0, C_in, BLOCK_C):
            ci_offs = ci_start + tl.arange(0, BLOCK_C)
            ci_mask = ci_offs < C_in  # [BLOCK_C]

            # We need to compute, for each of the 4 output positions in 2x2 pool window,
            # the 3x3 conv: sum_{kh,kw} input[n, ci, h+kh-1, w+kw-1] * w_flipped[ci, oc, kh, kw]
            # where w_flipped[ci, oc, kh, kw] = original_weight[ci, oc, 2-kh, 2-kw]
            # For ConvTranspose2d with stride=1,pad=1,k=3, output[h,w] = sum_{kh,kw,ci} x[ci, h-kh+1, w-kw+1] * W[ci, oc, kh, kw]
            # Let kh'=2-kh, kw'=2-kw: output[h,w] = sum_{kh',kw',ci} x[ci, h+kh'-1, w+kw'-1] * W[ci, oc, 2-kh', 2-kw']
            # So flipped weight at (kh', kw') is W[ci, oc, 2-kh', 2-kw'].

            # Load 9 weight values for each ci in block, shape [BLOCK_C]
            # weight pointer: w_ptr + ci*(C_out*9) + oc*9 + (2-kh)*3 + (2-kw)
            w_base = ci_offs * (C_out * 9) + oc * 9  # [BLOCK_C]

            # For each of the 9 kernel positions, we'll handle separately
            # kh, kw in {0,1,2}, flipped index = 2-kh, 2-kw
            # We'll unroll explicitly

            # Pre-load all 9 weights per ci
            w00 = tl.load(w_ptr + w_base + (2 * 3 + 2), mask=ci_mask, other=0.0)  # kh=0,kw=0 -> flipped (2,2)
            w01 = tl.load(w_ptr + w_base + (2 * 3 + 1), mask=ci_mask, other=0.0)  # kh=0,kw=1 -> (2,1)
            w02 = tl.load(w_ptr + w_base + (2 * 3 + 0), mask=ci_mask, other=0.0)
            w10 = tl.load(w_ptr + w_base + (1 * 3 + 2), mask=ci_mask, other=0.0)
            w11 = tl.load(w_ptr + w_base + (1 * 3 + 1), mask=ci_mask, other=0.0)
            w12 = tl.load(w_ptr + w_base + (1 * 3 + 0), mask=ci_mask, other=0.0)
            w20 = tl.load(w_ptr + w_base + (0 * 3 + 2), mask=ci_mask, other=0.0)
            w21 = tl.load(w_ptr + w_base + (0 * 3 + 1), mask=ci_mask, other=0.0)
            w22 = tl.load(w_ptr + w_base + (0 * 3 + 0), mask=ci_mask, other=0.0)

            # For each of 4 output positions (h0+dh, w0+dw), dh,dw in {0,1}
            # Conv: sum over kh,kw in {0,1,2}, ci of x[ci, h+kh-1, w+kw-1] * w[kh,kw]
            # We need to load x at all needed positions. Each output needs 3x3 = 9 input positions.
            # Adjacent outputs share input positions. Specifically:
            # out at (h, w) uses input rows h-1, h, h+1 and cols w-1, w, w+1
            # out at (h, w+1) uses cols w, w+1, w+2
            # out at (h+1, w) uses rows h, h+1, h+2
            # out at (h+1, w+1) uses rows h, h+1, h+2 and cols w, w+1, w+2
            # Total unique: rows h-1, h, h+1, h+2; cols w-1, w, w+1, w+2 -> 4x4 = 16 positions

            # Compute input base per ci
            # input offset: in_base_n + ci * in_HW + row * W + col
            # We'll loop over the 16 input positions, accumulating into the 4 outputs.

            # Actually simpler: for each output position separately, load 9 inputs and multiply.
            # But 4 outputs * 9 loads = 36 loads; with sharing = 16. Let's do the 16-load version.

            # However, with BLOCK_P positions, each "load" is actually a gather across BLOCK_P
            # positions with different h0, w0. So they aren't truly shared across BLOCK_P.
            # But they ARE shared across the 4 output positions WITHIN each pool window.
            # So 16 loads per (BLOCK_P, BLOCK_C) tile is correct.

            # Define rows and cols
            # rows: h0-1, h0, h0+1, h0+2
            # cols: w0-1, w0, w0+1, w0+2

            # ci dimension: [BLOCK_C, 1] when combined with BLOCK_P
            # We use [BLOCK_C, BLOCK_P] tiles via outer product.

            # ci offset broadcasted: [BLOCK_C, 1]
            ci_off_2d = ci_offs[:, None] * in_HW  # [BLOCK_C, 1]
            ci_mask_2d = ci_mask[:, None]  # [BLOCK_C, 1]

            # spatial mask per position [1, BLOCK_P]
            mask_p_2d = mask_p[None, :]  # [1, BLOCK_P]

            def load_input(rr, cc, rr_valid, cc_valid):
                # rr, cc: [BLOCK_P]
                # rr_valid, cc_valid: [BLOCK_P] bool
                spatial_off = rr[None, :] * W + cc[None, :]  # [1, BLOCK_P]
                idx = in_base_n + ci_off_2d + spatial_off  # [BLOCK_C, BLOCK_P]
                valid = ci_mask_2d & mask_p_2d & (rr_valid & cc_valid)[None, :]
                return tl.load(x_ptr + idx, mask=valid, other=0.0)

            # 16 input positions
            r0 = h0 - 1
            r1 = h0
            r2 = h0 + 1
            r3 = h0 + 2
            c0 = w0 - 1
            c1 = w0
            c2 = w0 + 1
            c3 = w0 + 2

            r0v = (r0 >= 0) & (r0 < H)
            r1v = (r1 >= 0) & (r1 < H)
            r2v = (r2 >= 0) & (r2 < H)
            r3v = (r3 >= 0) & (r3 < H)
            c0v = (c0 >= 0) & (c0 < W)
            c1v = (c1 >= 0) & (c1 < W)
            c2v = (c2 >= 0) & (c2 < W)
            c3v = (c3 >= 0) & (c3 < W)

            x00 = load_input(r0, c0, r0v, c0v)
            x01 = load_input(r0, c1, r0v, c1v)
            x02 = load_input(r0, c2, r0v, c2v)
            x03 = load_input(r0, c3, r0v, c3v)
            x10 = load_input(r1, c0, r1v, c0v)
            x11 = load_input(r1, c1, r1v, c1v)
            x12 = load_input(r1, c2, r1v, c2v)
            x13 = load_input(r1, c3, r1v, c3v)
            x20 = load_input(r2, c0, r2v, c0v)
            x21 = load_input(r2, c1, r2v, c1v)
            x22 = load_input(r2, c2, r2v, c2v)
            x23 = load_input(r2, c3, r2v, c3v)
            x30 = load_input(r3, c0, r3v, c0v)
            x31 = load_input(r3, c1, r3v, c1v)
            x32 = load_input(r3, c2, r3v, c2v)
            x33 = load_input(r3, c3, r3v, c3v)

            # weights broadcast to [BLOCK_C, 1]
            w00b = w00[:, None]
            w01b = w01[:, None]
            w02b = w02[:, None]
            w10b = w10[:, None]
            w11b = w11[:, None]
            w12b = w12[:, None]
            w20b = w20[:, None]
            w21b = w21[:, None]
            w22b = w22[:, None]

            # Output (h0, w0) = x[h0-1,w0-1]*w00 + x[h0-1,w0]*w01 + x[h0-1,w0+1]*w02
            #                 + x[h0,w0-1]*w10   + x[h0,w0]*w11   + x[h0,w0+1]*w12
            #                 + x[h0+1,w0-1]*w20 + x[h0+1,w0]*w21 + x[h0+1,w0+1]*w22
            o00 = (x00*w00b + x01*w01b + x02*w02b +
                   x10*w10b + x11*w11b + x12*w12b +
                   x20*w20b + x21*w21b + x22*w22b)

            # Output (h0, w0+1): uses cols c1, c2, c3
            o01 = (x01*w00b + x02*w01b + x03*w02b +
                   x11*w10b + x12*w11b + x13*w12b +
                   x21*w20b + x22*w21b + x23*w22b)

            # Output (h0+1, w0): uses rows r1, r2, r3
            o10 = (x10*w00b + x11*w01b + x12*w02b +
                   x20*w10b + x21*w11b + x22*w12b +
                   x30*w20b + x31*w21b + x32*w22b)

            # Output (h0+1, w0+1)
            o11 = (x11*w00b + x12*w01b + x13*w02b +
                   x21*w10b + x22*w11b + x23*w12b +
                   x31*w20b + x32*w21b + x33*w22b)

            # Sum over ci
            conv00 += tl.sum(o00, axis=0)
            conv01 += tl.sum(o01, axis=0)
            conv10 += tl.sum(o10, axis=0)
            conv11 += tl.sum(o11, axis=0)

        # Maxpool over 2x2
        m = tl.maximum(tl.maximum(conv00, conv01), tl.maximum(conv10, conv11))
        # Hardtanh
        m = tl.minimum(tl.maximum(m, hardtanh_min), hardtanh_max)
        m = tl.where(mask_p, m, 0.0)
        acc_sum += tl.sum(m, axis=0)

    mean_val = acc_sum * inv_count
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = float(hardtanh_min)
        self.hardtanh_max = float(hardtanh_max)
        # Only the specific config is fused; check below
        self._fused_ok = (kernel_size == 3 and stride == 1 and padding == 1 and
                         maxpool_kernel_size == 2 and maxpool_stride == 2)

    def forward(self, x):
        if not self._fused_ok:
            x = self.conv_transpose(x)
            x = F.max_pool2d(x, self.maxpool_kernel_size, self.maxpool_stride)
            x = F.hardtanh(x, self.hardtanh_min, self.hardtanh_max)
            x = torch.mean(x, dim=(2, 3), keepdim=True)
            return torch.tanh(x)

        x = x.contiguous()
        N, C_in, H, W = x.shape
        C_out = self.out_channels
        # Output of ConvTranspose2d with k=3,s=1,p=1 has same H,W as input
        pooled_H = H // 2
        pooled_W = W // 2
        total = pooled_H * pooled_W

        out = torch.empty((N, C_out, 1, 1), device=x.device, dtype=x.dtype)

        weight = self.conv_transpose.weight.contiguous()  # (C_in, C_out, 3, 3)
        bias = self.conv_transpose.bias
        if bias is None:
            bias = torch.zeros(C_out, device=x.device, dtype=x.dtype)
        bias = bias.contiguous()

        BLOCK_P = 64
        BLOCK_C = 16

        grid = (N * C_out,)
        fused_conv_pool_htanh_mean_tanh_kernel[grid](
            x, weight, bias, out,
            N, C_in, C_out, H, W,
            pooled_H, pooled_W,
            self.hardtanh_min, self.hardtanh_max,
            1.0 / float(total),
            BLOCK_P=BLOCK_P,
            BLOCK_C=BLOCK_C,
            num_warps=4,
            num_stages=2,
        )
        return out