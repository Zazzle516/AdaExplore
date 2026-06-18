import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ConvTranspose2d with k=3, stride=1, padding=1 is equivalent to a regular conv2d
# with the spatially-flipped weight and padding=1 (output H,W = input H,W).
# We fuse: conv + 2x2 maxpool + hardtanh + spatial mean into a single kernel.
# Each program computes one (n, oc) output by sweeping over output spatial tiles
# of size (BH, BW) where each output element is the max over a 2x2 conv-output window.
# We accumulate the hardtanh(max) values into a scalar sum, divide by count, tanh, store.

@triton.jit
def fused_convt_pool_htanh_mean_tanh_kernel(
    x_ptr,         # [N, IC, H, W]
    w_ptr,         # [IC, OC, 3, 3] (ConvTranspose2d weight layout)
    b_ptr,         # [OC]
    out_ptr,       # [N, OC, 1, 1]
    N, IC, OC, H, W,
    inv_count,
    HTANH_MIN: tl.constexpr,
    HTANH_MAX: tl.constexpr,
    BH: tl.constexpr,   # output tile height (in pooled coords)
    BW: tl.constexpr,   # output tile width  (in pooled coords)
    IC_BLOCK: tl.constexpr,
):
    # one program per (n, oc, tile)
    pid = tl.program_id(0)
    n = tl.program_id(1)
    oc = tl.program_id(2)

    H_out = H // 2
    W_out = W // 2
    tiles_w = (W_out + BW - 1) // BW
    tile_h = pid // tiles_w
    tile_w = pid % tiles_w

    oh_start = tile_h * BH  # in pooled coords
    ow_start = tile_w * BW

    # conv-output (pre-pool) coords for this tile: 2*BH x 2*BW
    # ih in [oh_start*2, oh_start*2 + 2*BH)
    # iw in [ow_start*2, ow_start*2 + 2*BW)
    H2 = BH * 2
    W2 = BW * 2

    # Coordinates for conv output positions in this tile
    rh = tl.arange(0, H2)  # [H2]
    rw = tl.arange(0, W2)  # [W2]
    ih = oh_start * 2 + rh  # [H2]
    iw = ow_start * 2 + rw  # [W2]
    ih_mask = ih < H
    iw_mask = iw < W

    # Accumulator for conv output: [H2, W2]
    conv_acc = tl.zeros([H2, W2], dtype=tl.float32)

    # ConvTranspose2d equivalent: y[oc, ih, iw] = sum_{ic, kh, kw} x[ic, ih + kh - 1, iw + kw - 1] * w[ic, oc, 2-kh, 2-kw]
    # => y[oc, ih, iw] = sum_{ic, dh, dw} x[ic, ih+dh, iw+dw] * w_flip[ic, oc, dh+1, dw+1]
    # where dh, dw in {-1, 0, 1}
    # i.e. flipped weight wf[ic, oc, a, b] = w[ic, oc, 2-a, 2-b], conv with padding=1.

    for ic_base in range(0, IC, IC_BLOCK):
        ic_offs = ic_base + tl.arange(0, IC_BLOCK)  # [IC_BLOCK]
        ic_mask = ic_offs < IC

        # Load weight tile: wf[ic, a, b] for fixed oc -> shape [IC_BLOCK, 3, 3]
        # w layout: [IC, OC, 3, 3], stride: (OC*9, 9, 3, 1)
        # wf[ic, a, b] = w[ic, oc, 2-a, 2-b]
        # We'll handle 3x3 by manual unroll over kh, kw

        for kh in tl.static_range(0, 3):
            for kw in tl.static_range(0, 3):
                # corresponding dh, dw
                dh = kh - 1
                dw = kw - 1
                # weight access: w[ic, oc, 2-kh, 2-kw]
                w_off = ic_offs * (OC * 9) + oc * 9 + (2 - kh) * 3 + (2 - kw)
                w_vals = tl.load(w_ptr + w_off, mask=ic_mask, other=0.0)  # [IC_BLOCK]

                # input positions: ih + dh, iw + dw
                ih_in = ih + dh  # [H2]
                iw_in = iw + dw  # [W2]
                ih_in_mask = (ih_in >= 0) & (ih_in < H)
                iw_in_mask = (iw_in >= 0) & (iw_in < W)

                # x[n, ic, ih_in, iw_in], shape [IC_BLOCK, H2, W2]
                # offset = n*IC*H*W + ic*H*W + ih_in*W + iw_in
                x_base = n * IC * H * W
                # broadcast: ic[IC_BLOCK,1,1], ih_in[1,H2,1], iw_in[1,1,W2]
                x_off = (x_base
                         + ic_offs[:, None, None] * (H * W)
                         + ih_in[None, :, None] * W
                         + iw_in[None, None, :])
                x_mask = (ic_mask[:, None, None]
                          & ih_in_mask[None, :, None]
                          & iw_in_mask[None, None, :])
                x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [IC_BLOCK, H2, W2]

                # multiply and accumulate
                conv_acc += tl.sum(x_vals * w_vals[:, None, None], axis=0)

    # Add bias
    bias_val = tl.load(b_ptr + oc)
    conv_acc = conv_acc + bias_val

    # 2x2 maxpool: reshape [H2, W2] -> [BH, 2, BW, 2] then max over axes 1, 3
    # Triton: do it via slicing using arange masks
    # We'll compute by doing 4 slices.
    # rh even/odd, rw even/odd
    # Build 4 [BH, BW] tensors
    rh_b = tl.arange(0, BH)
    rw_b = tl.arange(0, BW)

    # Mask for valid pooled outputs
    oh_idx = oh_start + rh_b  # [BH]
    ow_idx = ow_start + rw_b  # [BW]
    oh_valid = oh_idx < H_out
    ow_valid = ow_idx < W_out
    pool_mask = oh_valid[:, None] & ow_valid[None, :]

    # gather conv_acc[2*rh_b + a, 2*rw_b + b]
    # use direct indexing via reshape semantics: build flat indices into conv_acc
    # conv_acc shape [H2, W2]; flat idx = (2*rh_b+a)*W2 + (2*rw_b+b)
    # But we can't fancy-index a 2D triton tensor like that easily. Use tl.reshape.
    # Reshape [H2, W2] -> [BH, 2, BW, 2]
    pooled = tl.reshape(conv_acc, [BH, 2, BW, 2])
    m1 = tl.maximum(pooled[:, 0, :, 0], pooled[:, 0, :, 1])
    m2 = tl.maximum(pooled[:, 1, :, 0], pooled[:, 1, :, 1])
    pooled_max = tl.maximum(m1, m2)  # [BH, BW]

    # hardtanh
    pooled_max = tl.minimum(tl.maximum(pooled_max, HTANH_MIN), HTANH_MAX)
    pooled_max = tl.where(pool_mask, pooled_max, 0.0)

    # accumulate sum for this tile
    tile_sum = tl.sum(tl.sum(pooled_max, axis=1), axis=0)

    # Atomic add into output (which we'll later finalize with tanh)
    out_off = n * OC + oc
    tl.atomic_add(out_ptr + out_off, tile_sum)


@triton.jit
def finalize_kernel(buf_ptr, out_ptr, n_elem, inv_count):
    pid = tl.program_id(0)
    BLOCK: tl.constexpr = 256
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    s = tl.load(buf_ptr + offs, mask=mask, other=0.0)
    mean = s * inv_count
    e2 = tl.exp(2.0 * mean)
    out_val = (e2 - 1.0) / (e2 + 1.0)
    tl.store(out_ptr + offs, out_val, mask=mask)


@triton.jit
def fused_pool_htanh_mean_kernel(
    x_ptr,
    out_ptr,
    C, H, W,
    H_out, W_out,
    inv_count,
    HTANH_MIN: tl.constexpr,
    HTANH_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    base = n * C * H * W + c * H * W
    total = H_out * W_out

    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)

    for start in range(0, total, BLOCK):
        idx = start + offs
        mask = idx < total
        oh = idx // W_out
        ow = idx % W_out

        ih0 = oh * 2
        iw0 = ow * 2

        row0 = base + ih0 * W + iw0
        row1 = row0 + W

        v00 = tl.load(x_ptr + row0, mask=mask, other=-float('inf'))
        v01 = tl.load(x_ptr + row0 + 1, mask=mask, other=-float('inf'))
        v10 = tl.load(x_ptr + row1, mask=mask, other=-float('inf'))
        v11 = tl.load(x_ptr + row1 + 1, mask=mask, other=-float('inf'))

        m = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
        m = tl.minimum(tl.maximum(m, HTANH_MIN), HTANH_MAX)
        m = tl.where(mask, m, 0.0)
        acc += m

    s = tl.sum(acc, axis=0)
    mean = s * inv_count
    e2 = tl.exp(2.0 * mean)
    out_val = (e2 - 1.0) / (e2 + 1.0)
    tl.store(out_ptr + pid, out_val)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
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

    def forward(self, x):
        # Use the simple fused pool+htanh+mean+tanh kernel after the standard convT
        # which has been the fastest baseline.
        x = self.conv_transpose(x)
        x = x.contiguous()
        N, C, H, W = x.shape

        if self.maxpool_kernel_size == 2 and self.maxpool_stride == 2 and H % 2 == 0 and W % 2 == 0:
            H_out = H // 2
            W_out = W // 2
            out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
            inv_count = 1.0 / (H_out * W_out)
            BLOCK = 2048
            grid = (N * C,)
            fused_pool_htanh_mean_kernel[grid](
                x, out,
                C, H, W,
                H_out, W_out,
                inv_count,
                self.hardtanh_min, self.hardtanh_max,
                BLOCK=BLOCK,
                num_warps=8,
                num_stages=3,
            )
            return out
        else:
            x = F.max_pool2d(x, kernel_size=self.maxpool_kernel_size, stride=self.maxpool_stride)
            x = F.hardtanh(x, min_val=self.hardtanh_min, max_val=self.hardtanh_max)
            x = torch.mean(x, dim=(2, 3), keepdim=True)
            x = torch.tanh(x)
            return x