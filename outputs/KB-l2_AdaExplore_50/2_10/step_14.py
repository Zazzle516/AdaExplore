import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Fused: conv_transpose (stride=1, pad=1, k=3) implemented as conv2d with flipped+transposed weight,
# and tail (maxpool 2x2 + hardtanh + mean over spatial + tanh) all fused.
# Strategy: split work along output spatial. Each program handles one (n, oc) and a tile of
# pooled output positions. After conv+pool+hardtanh, accumulate into a partial-sum buffer
# atomically (or per-program then reduce). We use one program per (n, oc) doing the whole
# spatial reduction to avoid atomics.

@triton.jit
def fused_convt_pool_htanh_mean_tanh_kernel(
    x_ptr,        # input: [N, IC, H, W] (input to convT)
    w_ptr,        # weight (already prepared as conv weight): [OC, IC, KH, KW]
    b_ptr,        # bias: [OC]
    out_ptr,      # output: [N, OC, 1, 1]
    N, IC, H, W,
    OC,
    H_out, W_out,            # H, W of conv output (= H, W since stride=1 pad=1 k=3)
    Hp, Wp,                  # pooled dims = H_out//2, W_out//2
    inv_count,               # 1/(Hp*Wp)
    HTANH_MIN: tl.constexpr,
    HTANH_MAX: tl.constexpr,
    BLOCK_P: tl.constexpr,   # tile of pooled positions per iteration
    IC_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid % OC

    total_p = Hp * Wp

    offs = tl.arange(0, BLOCK_P)
    acc = tl.zeros([BLOCK_P], dtype=tl.float32)

    # base pointers
    x_n_base = n * IC * H * W
    w_oc_base = oc * IC * 9  # KH*KW = 9

    bias_val = tl.load(b_ptr + oc)

    # Loop over pooled output positions
    for p_start in range(0, total_p, BLOCK_P):
        p_idx = p_start + offs
        p_mask = p_idx < total_p
        ph = p_idx // Wp
        pw = p_idx % Wp

        # 2x2 pool window in conv output coords
        oh0 = ph * 2
        ow0 = pw * 2

        # conv output positions (oh0,ow0), (oh0,ow0+1), (oh0+1,ow0), (oh0+1,ow0+1)
        # For conv with pad=1, k=3, stride=1: out[oh,ow] = sum_{kh,kw,ic} x[ic, oh-1+kh, ow-1+kw] * w[oc,ic,kh,kw]
        # Initialize 4 conv accumulators with bias
        c00 = tl.zeros([BLOCK_P], dtype=tl.float32) + bias_val
        c01 = tl.zeros([BLOCK_P], dtype=tl.float32) + bias_val
        c10 = tl.zeros([BLOCK_P], dtype=tl.float32) + bias_val
        c11 = tl.zeros([BLOCK_P], dtype=tl.float32) + bias_val

        # iterate over input channels
        for ic in range(0, IC):
            x_ic_base = x_n_base + ic * H * W
            w_ic_base = w_oc_base + ic * 9

            # Load 9 weights (scalars)
            w00 = tl.load(w_ptr + w_ic_base + 0)
            w01 = tl.load(w_ptr + w_ic_base + 1)
            w02 = tl.load(w_ptr + w_ic_base + 2)
            w10 = tl.load(w_ptr + w_ic_base + 3)
            w11 = tl.load(w_ptr + w_ic_base + 4)
            w12 = tl.load(w_ptr + w_ic_base + 5)
            w20 = tl.load(w_ptr + w_ic_base + 6)
            w21 = tl.load(w_ptr + w_ic_base + 7)
            w22 = tl.load(w_ptr + w_ic_base + 8)

            # We need x at rows oh0-1, oh0, oh0+1, oh0+2  (for 2x2 outputs covering rows oh0,oh0+1)
            # and cols ow0-1, ow0, ow0+1, ow0+2
            # That's a 4x4 patch per pool window.
            # Load each of the 16 values (with bounds check).

            # Row indices
            r0 = oh0 - 1
            r1 = oh0
            r2 = oh0 + 1
            r3 = oh0 + 2
            c0 = ow0 - 1
            c1 = ow0
            c2 = ow0 + 1
            c3 = ow0 + 2

            # masks for valid (within H, W)
            r0_ok = (r0 >= 0) & (r0 < H)
            r1_ok = (r1 >= 0) & (r1 < H)
            r2_ok = (r2 >= 0) & (r2 < H)
            r3_ok = (r3 >= 0) & (r3 < H)
            c0_ok = (c0 >= 0) & (c0 < W)
            c1_ok = (c1 >= 0) & (c1 < W)
            c2_ok = (c2 >= 0) & (c2 < W)
            c3_ok = (c3 >= 0) & (c3 < W)

            # Load 16 values
            def_mask = p_mask

            # row 0
            m = def_mask & r0_ok & c0_ok
            x00 = tl.load(x_ptr + x_ic_base + r0 * W + c0, mask=m, other=0.0)
            m = def_mask & r0_ok & c1_ok
            x01 = tl.load(x_ptr + x_ic_base + r0 * W + c1, mask=m, other=0.0)
            m = def_mask & r0_ok & c2_ok
            x02 = tl.load(x_ptr + x_ic_base + r0 * W + c2, mask=m, other=0.0)
            m = def_mask & r0_ok & c3_ok
            x03 = tl.load(x_ptr + x_ic_base + r0 * W + c3, mask=m, other=0.0)

            # row 1
            m = def_mask & r1_ok & c0_ok
            x10 = tl.load(x_ptr + x_ic_base + r1 * W + c0, mask=m, other=0.0)
            m = def_mask & r1_ok & c1_ok
            x11 = tl.load(x_ptr + x_ic_base + r1 * W + c1, mask=m, other=0.0)
            m = def_mask & r1_ok & c2_ok
            x12 = tl.load(x_ptr + x_ic_base + r1 * W + c2, mask=m, other=0.0)
            m = def_mask & r1_ok & c3_ok
            x13 = tl.load(x_ptr + x_ic_base + r1 * W + c3, mask=m, other=0.0)

            # row 2
            m = def_mask & r2_ok & c0_ok
            x20 = tl.load(x_ptr + x_ic_base + r2 * W + c0, mask=m, other=0.0)
            m = def_mask & r2_ok & c1_ok
            x21 = tl.load(x_ptr + x_ic_base + r2 * W + c1, mask=m, other=0.0)
            m = def_mask & r2_ok & c2_ok
            x22 = tl.load(x_ptr + x_ic_base + r2 * W + c2, mask=m, other=0.0)
            m = def_mask & r2_ok & c3_ok
            x23 = tl.load(x_ptr + x_ic_base + r2 * W + c3, mask=m, other=0.0)

            # row 3
            m = def_mask & r3_ok & c0_ok
            x30 = tl.load(x_ptr + x_ic_base + r3 * W + c0, mask=m, other=0.0)
            m = def_mask & r3_ok & c1_ok
            x31 = tl.load(x_ptr + x_ic_base + r3 * W + c1, mask=m, other=0.0)
            m = def_mask & r3_ok & c2_ok
            x32 = tl.load(x_ptr + x_ic_base + r3 * W + c2, mask=m, other=0.0)
            m = def_mask & r3_ok & c3_ok
            x33 = tl.load(x_ptr + x_ic_base + r3 * W + c3, mask=m, other=0.0)

            # Conv outputs:
            # c[oh,ow] = sum_{kh,kw} x[oh-1+kh, ow-1+kw] * w[kh,kw]
            # c00 -> out at (oh0, ow0): uses rows 0..2 (r0..r2), cols 0..2 (c0..c2)
            c00 += x00 * w00 + x01 * w01 + x02 * w02 \
                 + x10 * w10 + x11 * w11 + x12 * w12 \
                 + x20 * w20 + x21 * w21 + x22 * w22

            # c01 -> out at (oh0, ow0+1): rows 0..2, cols 1..3
            c01 += x01 * w00 + x02 * w01 + x03 * w02 \
                 + x11 * w10 + x12 * w11 + x13 * w12 \
                 + x21 * w20 + x22 * w21 + x23 * w22

            # c10 -> out at (oh0+1, ow0): rows 1..3, cols 0..2
            c10 += x10 * w00 + x11 * w01 + x12 * w02 \
                 + x20 * w10 + x21 * w11 + x22 * w12 \
                 + x30 * w20 + x31 * w21 + x32 * w22

            # c11 -> out at (oh0+1, ow0+1): rows 1..3, cols 1..3
            c11 += x11 * w00 + x12 * w01 + x13 * w02 \
                 + x21 * w10 + x22 * w11 + x23 * w12 \
                 + x31 * w20 + x32 * w21 + x33 * w22

        # maxpool over the 2x2
        m_val = tl.maximum(tl.maximum(c00, c01), tl.maximum(c10, c11))
        # hardtanh
        m_val = tl.minimum(tl.maximum(m_val, HTANH_MIN), HTANH_MAX)
        m_val = tl.where(p_mask, m_val, 0.0)
        acc += m_val

    s = tl.sum(acc, axis=0)
    mean = s * inv_count
    e2 = tl.exp(2.0 * mean)
    out_val = (e2 - 1.0) / (e2 + 1.0)
    tl.store(out_ptr + pid, out_val)


# Fallback fused tail kernel (used when fast convT path doesn't apply)
@triton.jit
def fused_pool_htanh_mean_kernel(
    x_ptr,
    out_ptr,
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

        # Cache prepared conv weight (flipped+transposed) for fast path
        # ConvTranspose2d weight shape: [in_channels, out_channels, kH, kW]
        # Equivalent conv2d weight: [out_channels, in_channels, kH, kW] with kernel flipped over (kH,kW)
        self._cached_weight_version = None
        self._cached_conv_weight = None

    def _get_conv_weight(self):
        w = self.conv_transpose.weight  # [IC, OC, kH, kW]
        ver = w._version
        if self._cached_weight_version != ver or self._cached_conv_weight is None:
            # Flip over spatial dims and swap IC<->OC
            cw = torch.flip(w, dims=[2, 3]).permute(1, 0, 2, 3).contiguous()
            self._cached_conv_weight = cw
            self._cached_weight_version = ver
        return self._cached_conv_weight

    def forward(self, x):
        # Fast path: stride=1, padding=1, kernel=3, maxpool 2x2 stride 2
        if (self.stride == 1 and self.padding == 1 and self.kernel_size == 3
                and self.maxpool_kernel_size == 2 and self.maxpool_stride == 2):
            x = x.contiguous().cuda()
            N, IC, H, W = x.shape
            OC = self.out_channels
            # conv output size = same as H, W
            H_out = H
            W_out = W
            if H_out % 2 == 0 and W_out % 2 == 0:
                Hp = H_out // 2
                Wp = W_out // 2

                conv_w = self._get_conv_weight()  # [OC, IC, 3, 3]
                bias = self.conv_transpose.bias
                if bias is None:
                    bias = torch.zeros(OC, device=x.device, dtype=x.dtype)
                bias = bias.contiguous()

                out = torch.empty((N, OC, 1, 1), device=x.device, dtype=x.dtype)
                inv_count = 1.0 / (Hp * Wp)

                BLOCK_P = 64
                grid = (N * OC,)
                fused_convt_pool_htanh_mean_tanh_kernel[grid](
                    x, conv_w, bias, out,
                    N, IC, H, W, OC,
                    H_out, W_out, Hp, Wp,
                    inv_count,
                    self.hardtanh_min, self.hardtanh_max,
                    BLOCK_P=BLOCK_P,
                    IC_BLOCK=1,
                    num_warps=4,
                    num_stages=2,
                )
                return out

        # Fallback
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
                N, C, H, W,
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