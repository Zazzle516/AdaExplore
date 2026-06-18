import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Fused conv_transpose (k=3,s=1,p=1) + maxpool 2x2 + hardtanh + mean + tanh.
# Since stride=1, padding=1, kernel=3: ConvTranspose2d is equivalent to
# regular Conv2d with the spatially-flipped weight and padding=1.
# We compute conv outputs in 2x2 tiles and immediately reduce (max, hardtanh, sum).

@triton.jit
def fused_convt_pool_htanh_mean_tanh_kernel(
    x_ptr,        # input [N, IC, H, W]
    w_ptr,        # weight [IC, OC, 3, 3] (ConvTranspose2d weight layout)
    b_ptr,        # bias [OC]
    out_ptr,      # output [N, OC, 1, 1]
    N, IC, OC, H, W,
    H_out, W_out, # H/2, W/2
    inv_count,
    HTANH_MIN: tl.constexpr,
    HTANH_MAX: tl.constexpr,
    BLOCK_SP: tl.constexpr,  # number of pooled output positions per program (along spatial)
    IC_BLOCK: tl.constexpr,
):
    # grid: (N * OC, num_sp_blocks)
    pid_nc = tl.program_id(0)
    pid_sp = tl.program_id(1)

    n = pid_nc // OC
    oc = pid_nc % OC

    total_sp = H_out * W_out
    sp_start = pid_sp * BLOCK_SP
    sp_offs = sp_start + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < total_sp

    oh = sp_offs // W_out  # output (pooled) row in [0, H_out)
    ow = sp_offs % W_out

    # 2x2 conv output positions (pre-pool) start at (oh*2, ow*2)
    h0 = oh * 2  # conv output rows (= input H since same size)
    w0 = ow * 2

    # Initialize accumulators for the 2x2 conv outputs
    acc00 = tl.zeros([BLOCK_SP], dtype=tl.float32)
    acc01 = tl.zeros([BLOCK_SP], dtype=tl.float32)
    acc10 = tl.zeros([BLOCK_SP], dtype=tl.float32)
    acc11 = tl.zeros([BLOCK_SP], dtype=tl.float32)

    # ConvTranspose2d with k=3,s=1,p=1 equivalent to Conv2d with flipped weight, p=1.
    # Output[n,oc,h,w] = sum_{ic, kh, kw} W[ic, oc, 2-kh, 2-kw] * X[n, ic, h+kh-1, w+kw-1] + b[oc]
    # i.e. equivalent kernel WC[ic,oc,kh,kw] = W[ic,oc,2-kh,2-kw], pad=1.

    # Loop over IC
    for ic_base in range(0, IC, IC_BLOCK):
        ic_offs = ic_base + tl.arange(0, IC_BLOCK)
        ic_mask = ic_offs < IC

        # Loop over kernel positions (kh, kw) - unrolled
        for kh in tl.static_range(0, 3):
            for kw in tl.static_range(0, 3):
                # WC[ic, oc, kh, kw] = W[ic, oc, 2-kh, 2-kw]
                # weight pointer
                w_idx = ic_offs * (OC * 9) + oc * 9 + (2 - kh) * 3 + (2 - kw)
                wv = tl.load(w_ptr + w_idx, mask=ic_mask, other=0.0)  # [IC_BLOCK]

                # For each output pixel in the 2x2 tile (dh, dw):
                # input row = h0 + dh + kh - 1
                # input col = w0 + dw + kw - 1
                # We need to compute for (dh,dw) in {(0,0),(0,1),(1,0),(1,1)}

                for dh in tl.static_range(0, 2):
                    ih = h0 + dh + kh - 1  # [BLOCK_SP]
                    h_in = (ih >= 0) & (ih < H)
                    for dw in tl.static_range(0, 2):
                        iw = w0 + dw + kw - 1
                        w_in = (iw >= 0) & (iw < W)
                        valid = h_in & w_in & sp_mask  # [BLOCK_SP]

                        # base index per spatial: n*IC*H*W + ih*W + iw
                        # need + ic*H*W per IC
                        base_sp = n * IC * H * W + ih * W + iw  # [BLOCK_SP]
                        # ptrs: [BLOCK_SP, IC_BLOCK]
                        x_ptrs = x_ptr + base_sp[:, None] + ic_offs[None, :] * (H * W)
                        m = valid[:, None] & ic_mask[None, :]
                        xv = tl.load(x_ptrs, mask=m, other=0.0)  # [BLOCK_SP, IC_BLOCK]

                        # multiply and accumulate
                        prod = xv * wv[None, :]
                        s = tl.sum(prod, axis=1)  # [BLOCK_SP]

                        if dh == 0 and dw == 0:
                            acc00 += s
                        if dh == 0 and dw == 1:
                            acc01 += s
                        if dh == 1 and dw == 0:
                            acc10 += s
                        if dh == 1 and dw == 1:
                            acc11 += s

    # Add bias
    bias = tl.load(b_ptr + oc)
    acc00 += bias
    acc01 += bias
    acc10 += bias
    acc11 += bias

    # Maxpool 2x2
    m = tl.maximum(tl.maximum(acc00, acc01), tl.maximum(acc10, acc11))
    # Hardtanh
    m = tl.minimum(tl.maximum(m, HTANH_MIN), HTANH_MAX)
    # mask out invalid positions
    m = tl.where(sp_mask, m, 0.0)

    # Partial sum for mean
    s = tl.sum(m, axis=0)
    partial = s * inv_count

    # atomic add into output
    tl.atomic_add(out_ptr + pid_nc, partial)


@triton.jit
def tanh_inplace_kernel(out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(out_ptr + offs, mask=mask, other=0.0)
    e2 = tl.exp(2.0 * x)
    y = (e2 - 1.0) / (e2 + 1.0)
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def fused_pool_htanh_mean_tanh_kernel(
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
        # Use post-conv-fused kernel (cuDNN conv + our fused tail)
        x = self.conv_transpose(x)
        x = x.contiguous()
        N, C, H, W = x.shape

        if self.maxpool_kernel_size == 2 and self.maxpool_stride == 2 and H % 2 == 0 and W % 2 == 0:
            H_out = H // 2
            W_out = W // 2
            out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
            inv_count = 1.0 / (H_out * W_out)
            BLOCK = 4096
            grid = (N * C,)
            fused_pool_htanh_mean_tanh_kernel[grid](
                x, out,
                C, H, W,
                H_out, W_out,
                inv_count,
                self.hardtanh_min, self.hardtanh_max,
                BLOCK=BLOCK,
                num_warps=4,
                num_stages=2,
            )
            return out
        else:
            x = F.max_pool2d(x, kernel_size=self.maxpool_kernel_size, stride=self.maxpool_stride)
            x = F.hardtanh(x, min_val=self.hardtanh_min, max_val=self.hardtanh_max)
            x = torch.mean(x, dim=(2, 3), keepdim=True)
            x = torch.tanh(x)
            return x