import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_pool_htanh_mean_tanh_kernel(
    x_ptr,        # input NHWC: [N, H, W, IC]
    w_ptr,        # weight [OC, IC, 3, 3] (already transposed equivalent)
    b_ptr,        # bias [OC]
    out_ptr,      # output [N, OC]
    N, H, W, IC, OC,
    H_pool, W_pool,
    htanh_min: tl.constexpr,
    htanh_max: tl.constexpr,
    inv_area,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    IC_C: tl.constexpr,
):
    # one program per (n, oc_block)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    total = H_pool * W_pool
    sp_offs = tl.arange(0, BLOCK_SP)
    ic_range = tl.arange(0, IC_C)

    # accumulator for sum over spatial (per OC) [BLOCK_OC]
    sum_acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Load bias once
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    # x base pointer for this n
    x_base = pid_n * H * W * IC

    for sp_start in range(0, total, BLOCK_SP):
        idx = sp_start + sp_offs  # [BLOCK_SP]
        sp_mask = idx < total
        ho = idx // W_pool
        wo = idx % W_pool
        # 2x2 maxpool: input positions are (2*ho..2*ho+1, 2*wo..2*wo+1)
        h0 = ho * 2
        w0 = wo * 2

        # We compute conv outputs for 4 positions (h0, w0), (h0, w0+1), (h0+1, w0), (h0+1, w0+1)
        # Conv output[oc, h, w] = sum_{ic, kh, kw} x[ic, h+kh-1, w+kw-1] * W[oc, ic, kh, kw] + bias
        # (since padding=1, kernel=3, stride=1)

        # Initialize 4 accumulators per (BLOCK_OC, BLOCK_SP)
        c00 = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)
        c01 = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)
        c10 = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)
        c11 = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

        # loop over kernel
        for kh in tl.static_range(0, 3):
            for kw in tl.static_range(0, 3):
                # weight slice: [BLOCK_OC, IC]
                # w_ptr[oc, ic, kh, kw] -> oc * (IC*9) + ic*9 + kh*3 + kw
                w_off = oc_offs[:, None] * (IC * 9) + ic_range[None, :] * 9 + kh * 3 + kw
                w_vals = tl.load(w_ptr + w_off, mask=oc_mask[:, None], other=0.0)  # [BLOCK_OC, IC]

                # For each of 4 positions, gather x
                # position (h0+i, w0+j) -> input pos (h0+i+kh-1, w0+j+kw-1)
                for ii in tl.static_range(0, 2):
                    for jj in tl.static_range(0, 2):
                        ih = h0 + ii + kh - 1  # [BLOCK_SP]
                        iw = w0 + jj + kw - 1  # [BLOCK_SP]
                        in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & sp_mask
                        # x offset: n*H*W*IC + ih*W*IC + iw*IC + ic
                        x_off = x_base + ih[:, None] * (W * IC) + iw[:, None] * IC + ic_range[None, :]
                        x_vals = tl.load(x_ptr + x_off, mask=in_bounds[:, None], other=0.0)  # [BLOCK_SP, IC]

                        # accumulate: [BLOCK_OC, IC] @ [IC, BLOCK_SP]
                        prod = tl.dot(w_vals, tl.trans(x_vals))  # [BLOCK_OC, BLOCK_SP]
                        if (ii == 0) and (jj == 0):
                            c00 += prod
                        if (ii == 0) and (jj == 1):
                            c01 += prod
                        if (ii == 1) and (jj == 0):
                            c10 += prod
                        if (ii == 1) and (jj == 1):
                            c11 += prod

        # add bias
        c00 += bias[:, None]
        c01 += bias[:, None]
        c10 += bias[:, None]
        c11 += bias[:, None]

        # maxpool 2x2
        m = tl.maximum(tl.maximum(c00, c01), tl.maximum(c10, c11))
        # hardtanh
        m = tl.minimum(tl.maximum(m, htanh_min), htanh_max)
        # mask invalid spatial
        m = tl.where(sp_mask[None, :], m, 0.0)
        # accumulate sum
        sum_acc += tl.sum(m, axis=1)

    mean_val = sum_acc * inv_area
    # tanh
    e2 = tl.exp(2.0 * mean_val)
    t = (e2 - 1.0) / (e2 + 1.0)

    out_off = pid_n * OC + oc_offs
    tl.store(out_ptr + out_off, t, mask=oc_mask)


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

        # Conditions for fast path: stride=1, padding=1, k=3
        self._fast = (stride == 1 and padding == 1 and kernel_size == 3
                      and maxpool_kernel_size == 2 and maxpool_stride == 2)

    def _get_equivalent_weight(self):
        # ConvTranspose2d with stride=1, padding=1, k=3 is equivalent to
        # Conv2d with padding=1 using weight that is:
        #   - transpose of dims (in, out) -> (out, in)
        #   - spatially flipped
        # ConvTranspose2d weight shape: [in_channels, out_channels, kH, kW]
        w = self.conv_transpose.weight  # [IC, OC, 3, 3]
        # transpose to [OC, IC, 3, 3] and flip spatial
        w_eq = w.permute(1, 0, 2, 3).contiguous()
        w_eq = torch.flip(w_eq, dims=(2, 3)).contiguous()
        return w_eq

    def forward(self, x):
        if not self._fast:
            x = self.conv_transpose(x)
            x = F.max_pool2d(x, self.maxpool_kernel_size, self.maxpool_stride)
            x = F.hardtanh(x, self.hardtanh_min, self.hardtanh_max)
            x = torch.mean(x, dim=(2, 3), keepdim=True)
            x = torch.tanh(x)
            return x

        N, IC, H, W = x.shape
        OC = self.out_channels

        # Convert input to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        w_eq = self._get_equivalent_weight()  # [OC, IC, 3, 3]
        bias = self.conv_transpose.bias

        H_pool = H // 2
        W_pool = W // 2

        out = torch.empty((N, OC, 1, 1), device=x.device, dtype=x.dtype)
        out_flat = out.view(N, OC)

        BLOCK_OC = 32
        BLOCK_SP = 64
        IC_C = IC  # must be power of 2 ideally; 64 works

        grid = (N, triton.cdiv(OC, BLOCK_OC))

        inv_area = 1.0 / float(H_pool * W_pool)

        fused_conv_pool_htanh_mean_tanh_kernel[grid](
            x_nhwc, w_eq, bias, out_flat,
            N, H, W, IC, OC,
            H_pool, W_pool,
            self.hardtanh_min, self.hardtanh_max,
            inv_area,
            BLOCK_OC=BLOCK_OC,
            BLOCK_SP=BLOCK_SP,
            IC_C=IC_C,
            num_warps=4,
            num_stages=2,
        )
        return out