import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ConvTranspose2d with kernel=3, stride=1, padding=1 is mathematically equivalent
# to a regular Conv2d with the spatially-flipped weight (and same bias).
# Output: same H,W as input. We then do 2x2 maxpool, hardtanh, mean, tanh.
#
# Strategy: fuse conv + maxpool(2x2) + hardtanh + mean over (H/2, W/2) + tanh
# into a single Triton kernel.
#
# Each program computes one (n, c_out) output value:
#   For each pooled output position (oh, ow) in [0, H/2) x [0, W/2):
#       For each of the 4 conv positions in the 2x2 window at (2*oh + dh, 2*ow + dw):
#           accumulate conv result over IC * 3 * 3
#       Take max over 4 -> hardtanh -> add to running sum
#   Divide by (H/2 * W/2) -> tanh -> store
#
# The loop nest is heavy; instead we iterate spatially in tiles inside the kernel.
# To keep it simple and fast, we use a tiled approach: each program handles
# one (n, c_out) and iterates over the spatial grid.

@triton.jit
def fused_convT_pool_htanh_mean_tanh_kernel(
    x_ptr,            # [N, IC, H, W]
    w_ptr,            # [IC, OC, 3, 3]  (ConvTranspose2d weight layout)
    b_ptr,            # [OC]
    out_ptr,          # [N, OC, 1, 1]
    N, IC, OC, H, W,
    inv_count,        # 1.0 / (H_out * W_out) where H_out=H/2, W_out=W/2
    HTANH_MIN: tl.constexpr,
    HTANH_MAX: tl.constexpr,
    KH: tl.constexpr, # 3
    KW: tl.constexpr, # 3
    PAD: tl.constexpr, # 1
):
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid % OC

    H_out = H // 2
    W_out = W // 2

    # We'll iterate (oh, ow) over pooled output grid.
    # For each, compute 4 conv positions (2*oh+dh, 2*ow+dw) for dh,dw in {0,1}.
    # Conv: out[h,w] = sum_{ic} sum_{kh, kw} x[ic, h+kh-PAD, w+kw-PAD] * Wt[ic, oc, kh, kw]
    # (using regular conv with the ConvTranspose weight flipped: flipped index is (KH-1-kh, KW-1-kw))
    # For ConvTranspose2d output: y[h,w] = sum_ic sum_kh sum_kw x[ic, h+kh-PAD, w+kw-PAD] * W[ic, oc, KH-1-kh, KW-1-kw]
    # Equivalently: y[h,w] = sum_ic sum_kh sum_kw x[ic, h-kh+PAD, w-kw+PAD] * W[ic, oc, kh, kw]
    # We'll use the second form (matches ConvTranspose2d formula directly):
    #   in_h = h - kh + PAD ;  in_w = w - kw + PAD

    bias = tl.load(b_ptr + oc)

    acc_sum = tl.zeros([1], dtype=tl.float32)

    # Loop over pooled spatial positions
    for oh in range(0, H_out):
        for ow in range(0, W_out):
            # 4 conv positions
            # Compute each as a scalar accumulation
            best = tl.full([1], -float('inf'), dtype=tl.float32)
            for dh in tl.static_range(0, 2):
                for dw in tl.static_range(0, 2):
                    h = oh * 2 + dh
                    w = ow * 2 + dw
                    v = bias
                    # conv accumulation
                    for kh in tl.static_range(0, KH):
                        ih = h - kh + PAD
                        for kw in tl.static_range(0, KW):
                            iw = w - kw + PAD
                            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                            for ic in range(0, IC):
                                x_off = n * IC * H * W + ic * H * W + ih * W + iw
                                w_off = ic * OC * KH * KW + oc * KH * KW + kh * KW + kw
                                xv = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0)
                                wv = tl.load(w_ptr + w_off)
                                v += xv * wv
                    # update best (max)
                    best = tl.maximum(best, v)
            # hardtanh
            best = tl.minimum(tl.maximum(best, HTANH_MIN), HTANH_MAX)
            acc_sum += best

    mean = acc_sum * inv_count
    out_val = (tl.exp(2.0 * mean) - 1.0) / (tl.exp(2.0 * mean) + 1.0)
    tl.store(out_ptr + pid, tl.sum(out_val, axis=0))


# The fully-fused kernel above is too slow due to per-element load patterns.
# Instead, we use a hybrid: leverage torch's highly optimized convolution
# (via cudnn) for ConvTranspose2d, and fuse the post-conv chain
# (maxpool + hardtanh + mean + tanh) into a single Triton kernel.

@triton.jit
def fused_pool_htanh_mean_tanh_kernel(
    x_ptr,        # [N, C, H, W]
    out_ptr,      # [N, C, 1, 1]
    N, C, H, W,
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

        p00 = base + ih0 * W + iw0
        p01 = p00 + 1
        p10 = p00 + W
        p11 = p10 + 1

        v00 = tl.load(x_ptr + p00, mask=mask, other=-float('inf'))
        v01 = tl.load(x_ptr + p01, mask=mask, other=-float('inf'))
        v10 = tl.load(x_ptr + p10, mask=mask, other=-float('inf'))
        v11 = tl.load(x_ptr + p11, mask=mask, other=-float('inf'))

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
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = float(hardtanh_min)
        self.hardtanh_max = float(hardtanh_max)

        # If stride==1 and kernel_size==3, we can convert ConvTranspose2d into a
        # regular Conv2d with flipped weights. With padding=1 the spatial output
        # size matches the input. We precompute the equivalent conv weight at
        # init to leverage cudnn's fast forward conv path.
        self._can_use_conv = (stride == 1)
        if self._can_use_conv:
            with torch.no_grad():
                # ConvTranspose2d weight: [in_channels, out_channels, KH, KW]
                # Equivalent Conv2d weight: [out_channels, in_channels, KH, KW]
                # with spatial flip of kernel.
                w = self.conv_transpose.weight  # [IC, OC, KH, KW]
                w_flipped = torch.flip(w, dims=[2, 3]).permute(1, 0, 2, 3).contiguous()
                self._conv_weight = nn.Parameter(w_flipped, requires_grad=False)
                self._conv_bias = self.conv_transpose.bias  # share

    def _equivalent_conv(self, x):
        # Recompute equivalent weight from current ConvTranspose params
        # (in case weights changed). For inference this is one-time-ish.
        w = self.conv_transpose.weight
        w_eq = torch.flip(w, dims=[2, 3]).permute(1, 0, 2, 3).contiguous()
        return F.conv2d(x, w_eq, bias=self.conv_transpose.bias,
                        stride=1, padding=self.padding)

    def forward(self, x):
        if self._can_use_conv and self.kernel_size == 3:
            y = self._equivalent_conv(x)
        else:
            y = self.conv_transpose(x)
        y = y.contiguous()
        N, C, H, W = y.shape

        if (self.maxpool_kernel_size == 2 and self.maxpool_stride == 2
                and H % 2 == 0 and W % 2 == 0):
            H_out = H // 2
            W_out = W // 2
            out = torch.empty((N, C, 1, 1), device=y.device, dtype=y.dtype)
            inv_count = 1.0 / (H_out * W_out)
            BLOCK = 1024
            grid = (N * C,)
            fused_pool_htanh_mean_tanh_kernel[grid](
                y, out,
                N, C, H, W,
                H_out, W_out,
                inv_count,
                self.hardtanh_min, self.hardtanh_max,
                BLOCK=BLOCK,
                num_warps=4,
            )
            return out
        else:
            y = F.max_pool2d(y, kernel_size=self.maxpool_kernel_size, stride=self.maxpool_stride)
            y = F.hardtanh(y, min_val=self.hardtanh_min, max_val=self.hardtanh_max)
            y = torch.mean(y, dim=(2, 3), keepdim=True)
            y = torch.tanh(y)
            return y