import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_pool_htanh_mean_tanh_kernel(
    x_ptr,       # [N, IC, H, W]
    w_ptr,       # [OC, IC, 3, 3] -- already prepared as equivalent conv weight
    b_ptr,       # [OC]
    out_ptr,     # [N, OC]
    N, IC, H, W,
    H_out, W_out,  # pooled dims = H/2, W/2
    htanh_min, htanh_max,
    inv_area,
    BLOCK_IC: tl.constexpr,
    BLOCK_SP: tl.constexpr,  # number of pooled output positions per iteration
):
    # one program per (n, oc)
    pid = tl.program_id(0)
    n = pid // tl.num_programs(1)  # not used; use direct mapping
    # use 2D grid actually
    pass


@triton.jit
def fused_kernel(
    x_ptr,       # [N, IC, H, W]
    w_ptr,       # [OC, IC, 3, 3]
    b_ptr,       # [OC]
    out_ptr,     # [N, OC]
    N, IC, OC, H, W,
    H_out, W_out,
    htanh_min, htanh_max,
    inv_area,
    BLOCK_IC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    n = tl.program_id(0)
    oc = tl.program_id(1)

    total = H_out * W_out

    # load bias
    bias = tl.load(b_ptr + oc)

    # Load weights for this oc: shape [IC, 3, 3]. We'll load in chunks of BLOCK_IC.
    sum_acc = 0.0

    sp_offs = tl.arange(0, BLOCK_SP)

    for sp_start in range(0, total, BLOCK_SP):
        idx = sp_start + sp_offs
        mask_sp = idx < total
        ho = idx // W_out  # [BLOCK_SP]
        wo = idx % W_out
        # 4 conv output positions per pooled position: (h0..h0+1, w0..w0+1) where h0=2*ho
        h0 = ho * 2
        w0 = wo * 2

        # accumulators for 4 conv outputs
        acc00 = tl.zeros((BLOCK_SP,), dtype=tl.float32)
        acc01 = tl.zeros((BLOCK_SP,), dtype=tl.float32)
        acc10 = tl.zeros((BLOCK_SP,), dtype=tl.float32)
        acc11 = tl.zeros((BLOCK_SP,), dtype=tl.float32)

        # iterate over input channels
        for ic_start in range(0, IC, BLOCK_IC):
            ic_offs = ic_start + tl.arange(0, BLOCK_IC)
            ic_mask = ic_offs < IC  # [BLOCK_IC]

            # Load weights w[oc, ic_offs, kh, kw] for kh,kw in 0..2
            # weight stride: oc*IC*9 + ic*9 + kh*3 + kw
            w_base = oc * IC * 9 + ic_offs * 9  # [BLOCK_IC]

            w00 = tl.load(w_ptr + w_base + 0, mask=ic_mask, other=0.0)
            w01 = tl.load(w_ptr + w_base + 1, mask=ic_mask, other=0.0)
            w02 = tl.load(w_ptr + w_base + 2, mask=ic_mask, other=0.0)
            w10 = tl.load(w_ptr + w_base + 3, mask=ic_mask, other=0.0)
            w11 = tl.load(w_ptr + w_base + 4, mask=ic_mask, other=0.0)
            w12 = tl.load(w_ptr + w_base + 5, mask=ic_mask, other=0.0)
            w20 = tl.load(w_ptr + w_base + 6, mask=ic_mask, other=0.0)
            w21 = tl.load(w_ptr + w_base + 7, mask=ic_mask, other=0.0)
            w22 = tl.load(w_ptr + w_base + 8, mask=ic_mask, other=0.0)

            # For each of 4 conv outputs (dy, dx) in 0..1, accumulate
            # conv output at (oh, ow) = sum_{ic, kh, kw} x[ic, oh-1+kh, ow-1+kw] * w[ic,kh,kw]
            # We need x at positions row in {h0-1, h0, h0+1, h0+2} and col similarly.

            # For each input channel separately (loop over BLOCK_IC unrolled via tl ops):
            # We'll iterate kh, kw and accumulate into 4 conv outputs that touch (h0+dy, w0+dx).
            # For input row ih = h0+dy-1+kh, similarly col.
            # Easier: loop over the 4x4 patch of input rows (h0-1..h0+2) and cols (w0-1..w0+2).

            # Input base per ic: n*IC*H*W + ic*H*W
            x_base = n * IC * H * W + ic_offs * H * W  # [BLOCK_IC]

            # We need a 4x4 input patch per spatial position per ic, multiply with 3x3 weight,
            # accumulate into 2x2 outputs.
            # Loop over the 4 input rows and 4 input cols.
            for ir in tl.static_range(0, 4):
                ih = h0[:, None] - 1 + ir  # [BLOCK_SP, 1]
                row_valid = (ih >= 0) & (ih < H)
                for ic_ in tl.static_range(0, 4):
                    iw = w0[:, None] - 1 + ic_  # [BLOCK_SP, 1]
                    col_valid = (iw >= 0) & (iw < W)
                    in_mask = row_valid & col_valid & mask_sp[:, None] & ic_mask[None, :]
                    # input address: x_base[None,:] + ih*W + iw -> [BLOCK_SP, BLOCK_IC]
                    addr = x_base[None, :] + ih * W + iw
                    x_val = tl.load(x_ptr + addr, mask=in_mask, other=0.0)  # [BLOCK_SP, BLOCK_IC]

                    # contribute to conv outputs (dy, dx) where:
                    #   ih = h0+dy-1+kh => kh = ir - dy, must be in 0..2
                    #   iw = w0+dx-1+kw => kw = ic_ - dx, must be in 0..2
                    # For dy in {0,1}, kh = ir - dy. For dx in {0,1}, kw = ic_ - dx.
                    # Generate contributions for each (dy, dx) where 0 <= kh,kw <= 2.

                    # dy=0
                    kh0 = ir
                    if (kh0 >= 0) and (kh0 <= 2):
                        # dx=0
                        kw0 = ic_
                        if (kw0 >= 0) and (kw0 <= 2):
                            w_idx = kh0 * 3 + kw0
                            if w_idx == 0:
                                acc00 += tl.sum(x_val * w00[None, :], axis=1)
                            elif w_idx == 1:
                                acc00 += tl.sum(x_val * w01[None, :], axis=1)
                            elif w_idx == 2:
                                acc00 += tl.sum(x_val * w02[None, :], axis=1)
                            elif w_idx == 3:
                                acc00 += tl.sum(x_val * w10[None, :], axis=1)
                            elif w_idx == 4:
                                acc00 += tl.sum(x_val * w11[None, :], axis=1)
                            elif w_idx == 5:
                                acc00 += tl.sum(x_val * w12[None, :], axis=1)
                            elif w_idx == 6:
                                acc00 += tl.sum(x_val * w20[None, :], axis=1)
                            elif w_idx == 7:
                                acc00 += tl.sum(x_val * w21[None, :], axis=1)
                            elif w_idx == 8:
                                acc00 += tl.sum(x_val * w22[None, :], axis=1)
                        # dx=1
                        kw1 = ic_ - 1
                        if (kw1 >= 0) and (kw1 <= 2):
                            w_idx = kh0 * 3 + kw1
                            if w_idx == 0:
                                acc01 += tl.sum(x_val * w00[None, :], axis=1)
                            elif w_idx == 1:
                                acc01 += tl.sum(x_val * w01[None, :], axis=1)
                            elif w_idx == 2:
                                acc01 += tl.sum(x_val * w02[None, :], axis=1)
                            elif w_idx == 3:
                                acc01 += tl.sum(x_val * w10[None, :], axis=1)
                            elif w_idx == 4:
                                acc01 += tl.sum(x_val * w11[None, :], axis=1)
                            elif w_idx == 5:
                                acc01 += tl.sum(x_val * w12[None, :], axis=1)
                            elif w_idx == 6:
                                acc01 += tl.sum(x_val * w20[None, :], axis=1)
                            elif w_idx == 7:
                                acc01 += tl.sum(x_val * w21[None, :], axis=1)
                            elif w_idx == 8:
                                acc01 += tl.sum(x_val * w22[None, :], axis=1)
                    # dy=1
                    kh1 = ir - 1
                    if (kh1 >= 0) and (kh1 <= 2):
                        # dx=0
                        kw0 = ic_
                        if (kw0 >= 0) and (kw0 <= 2):
                            w_idx = kh1 * 3 + kw0
                            if w_idx == 0:
                                acc10 += tl.sum(x_val * w00[None, :], axis=1)
                            elif w_idx == 1:
                                acc10 += tl.sum(x_val * w01[None, :], axis=1)
                            elif w_idx == 2:
                                acc10 += tl.sum(x_val * w02[None, :], axis=1)
                            elif w_idx == 3:
                                acc10 += tl.sum(x_val * w10[None, :], axis=1)
                            elif w_idx == 4:
                                acc10 += tl.sum(x_val * w11[None, :], axis=1)
                            elif w_idx == 5:
                                acc10 += tl.sum(x_val * w12[None, :], axis=1)
                            elif w_idx == 6:
                                acc10 += tl.sum(x_val * w20[None, :], axis=1)
                            elif w_idx == 7:
                                acc10 += tl.sum(x_val * w21[None, :], axis=1)
                            elif w_idx == 8:
                                acc10 += tl.sum(x_val * w22[None, :], axis=1)
                        # dx=1
                        kw1 = ic_ - 1
                        if (kw1 >= 0) and (kw1 <= 2):
                            w_idx = kh1 * 3 + kw1
                            if w_idx == 0:
                                acc11 += tl.sum(x_val * w00[None, :], axis=1)
                            elif w_idx == 1:
                                acc11 += tl.sum(x_val * w01[None, :], axis=1)
                            elif w_idx == 2:
                                acc11 += tl.sum(x_val * w02[None, :], axis=1)
                            elif w_idx == 3:
                                acc11 += tl.sum(x_val * w10[None, :], axis=1)
                            elif w_idx == 4:
                                acc11 += tl.sum(x_val * w11[None, :], axis=1)
                            elif w_idx == 5:
                                acc11 += tl.sum(x_val * w12[None, :], axis=1)
                            elif w_idx == 6:
                                acc11 += tl.sum(x_val * w20[None, :], axis=1)
                            elif w_idx == 7:
                                acc11 += tl.sum(x_val * w21[None, :], axis=1)
                            elif w_idx == 8:
                                acc11 += tl.sum(x_val * w22[None, :], axis=1)

        # add bias
        acc00 = acc00 + bias
        acc01 = acc01 + bias
        acc10 = acc10 + bias
        acc11 = acc11 + bias

        # maxpool 2x2
        m = tl.maximum(tl.maximum(acc00, acc01), tl.maximum(acc10, acc11))
        # hardtanh
        m = tl.minimum(tl.maximum(m, htanh_min), htanh_max)

        sum_acc += tl.sum(tl.where(mask_sp, m, 0.0), axis=0)

    mean_val = sum_acc * inv_area
    e2 = tl.exp(2.0 * mean_val)
    t = (e2 - 1.0) / (e2 + 1.0)
    tl.store(out_ptr + n * OC + oc, t)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        # Keep the conv_transpose as the source of truth for parameters
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

        # Conditions for the fused path: stride=1, padding=1, kernel=3, maxpool 2x2 stride 2
        self.use_fused = (stride == 1 and padding == 1 and kernel_size == 3
                          and maxpool_kernel_size == 2 and maxpool_stride == 2)

    def _get_equiv_conv_weight(self):
        # ConvTranspose2d weight shape: [in_channels, out_channels, kH, kW]
        # Equivalent Conv2d weight (for stride=1, padding=1, k=3):
        # W_conv[oc, ic, kh, kw] = W_convT[ic, oc, kH-1-kh, kW-1-kw]
        w = self.conv_transpose.weight  # [IC, OC, 3, 3]
        w_eq = w.permute(1, 0, 2, 3).contiguous()  # [OC, IC, 3, 3]
        w_eq = torch.flip(w_eq, dims=(2, 3)).contiguous()
        return w_eq

    def forward(self, x):
        if not self.use_fused:
            x = self.conv_transpose(x)
            x = F.max_pool2d(x, self.maxpool_kernel_size, self.maxpool_stride)
            x = F.hardtanh(x, self.hardtanh_min, self.hardtanh_max)
            x = torch.mean(x, dim=(2, 3), keepdim=True)
            x = torch.tanh(x)
            return x

        x = x.contiguous()
        N, IC, H, W = x.shape
        OC = self.out_channels
        H_out = H // 2
        W_out = W // 2

        w_eq = self._get_equiv_conv_weight()
        bias = self.conv_transpose.bias.contiguous()

        out = torch.empty((N, OC), device=x.device, dtype=x.dtype)
        inv_area = 1.0 / float(H_out * W_out)

        grid = (N, OC)
        fused_kernel[grid](
            x, w_eq, bias, out,
            N, IC, OC, H, W,
            H_out, W_out,
            self.hardtanh_min, self.hardtanh_max,
            inv_area,
            BLOCK_IC=16,
            BLOCK_SP=64,
            num_warps=4,
        )
        return out.view(N, OC, 1, 1)