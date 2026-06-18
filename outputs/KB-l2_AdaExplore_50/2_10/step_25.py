import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ConvTranspose2d with stride=1, padding=1, kernel=3 is equivalent to
# Conv2d with weight spatially flipped (both H and W) and the IC/OC dims swapped.
# Original weight shape: (in_channels, out_channels, kH, kW)
# Equivalent conv weight: (out_channels, in_channels, kH, kW) with kH,kW flipped.


@triton.jit
def fused_conv_pool_htanh_mean_tanh_kernel(
    x_ptr,        # (B, IC, H, W) contiguous - input
    w_ptr,        # (OC, IC, 3, 3) contiguous - equivalent conv weight (flipped)
    b_ptr,        # (OC,) bias
    out_ptr,      # (B, OC) -> reshape to (B,OC,1,1)
    B, IC, H, W,
    OC,
    pooled_H, pooled_W,
    inv_count,
    hmin: tl.constexpr,
    hmax: tl.constexpr,
    BLOCK_HW: tl.constexpr,   # number of pooled positions per program
    IC_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)         # pooled-tile id along (pooled_H*pooled_W)/BLOCK_HW
    pid_nc = tl.program_id(1)      # (n, oc)

    n = pid_nc // OC
    oc = pid_nc % OC

    total = pooled_H * pooled_W

    offs = pid * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_p = offs < total

    ph = offs // pooled_W
    pw = offs - ph * pooled_W

    # 2x2 max pool over conv output -> output positions h0=ph*2..ph*2+1, w0=pw*2..pw*2+1
    # For each of 4 output positions (oh, ow), compute conv:
    #   sum over ic, kh, kw of x[n, ic, oh+kh-1, ow+kw-1] * w[oc, ic, kh, kw]
    # Bias added once per output.

    # Pre-compute h0, w0 (in conv-output coords)
    h0 = ph * 2
    w0 = pw * 2

    # Accumulators for the 4 output positions of the 2x2 window
    acc00 = tl.zeros((BLOCK_HW,), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_HW,), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_HW,), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_HW,), dtype=tl.float32)

    # Loop over input channels in blocks
    # weight ptr base for this oc: w_ptr + oc*IC*9
    w_oc_base = w_ptr + oc * IC * 9
    x_n_base = x_ptr + n * IC * H * W

    for ic_start in range(0, IC, IC_BLOCK):
        ic_offs = ic_start + tl.arange(0, IC_BLOCK)
        ic_mask = ic_offs < IC  # (IC_BLOCK,)

        # Load weights for this ic block: 9 values per ic
        # weight indexing: w[oc, ic, kh, kw] -> offset oc*IC*9 + ic*9 + kh*3 + kw
        w_base = w_oc_base + ic_offs * 9  # (IC_BLOCK,)

        w00 = tl.load(w_base + 0, mask=ic_mask, other=0.0)
        w01 = tl.load(w_base + 1, mask=ic_mask, other=0.0)
        w02 = tl.load(w_base + 2, mask=ic_mask, other=0.0)
        w10 = tl.load(w_base + 3, mask=ic_mask, other=0.0)
        w11 = tl.load(w_base + 4, mask=ic_mask, other=0.0)
        w12 = tl.load(w_base + 5, mask=ic_mask, other=0.0)
        w20 = tl.load(w_base + 6, mask=ic_mask, other=0.0)
        w21 = tl.load(w_base + 7, mask=ic_mask, other=0.0)
        w22 = tl.load(w_base + 8, mask=ic_mask, other=0.0)

        # For each output position (oh, ow) in the 2x2 window, we need a 3x3 input patch
        # centered at (oh, ow), i.e., input positions (oh-1..oh+1, ow-1..ow+1).
        # The 2x2 window covers oh in {h0, h0+1}, ow in {w0, w0+1}.
        # Combined input rows needed: h0-1, h0, h0+1, h0+2 (4 rows)
        # Combined input cols needed: w0-1, w0, w0+1, w0+2 (4 cols)
        # That's a 4x4 patch per (n, ic, ph, pw).

        # Compute the 16 input values for the 4x4 patch
        # For each (ic in IC_BLOCK, hw in BLOCK_HW), load x[n, ic, row, col]
        # x offset = x_n_base + ic*H*W + row*W + col

        # We'll handle out-of-bounds (padding) by mask -> 0.

        # base offset per ic per hw
        # shape: (IC_BLOCK, BLOCK_HW)
        x_ic_base = x_n_base + ic_offs[:, None] * (H * W)  # (IC_BLOCK, 1)

        # rows: r0=h0-1, r1=h0, r2=h0+1, r3=h0+2 (shape BLOCK_HW)
        r0 = h0 - 1
        r1 = h0
        r2 = h0 + 1
        r3 = h0 + 2
        c0 = w0 - 1
        c1 = w0
        c2 = w0 + 1
        c3 = w0 + 2

        r0_ok = (r0 >= 0) & (r0 < H)
        r1_ok = (r1 >= 0) & (r1 < H)
        r2_ok = (r2 >= 0) & (r2 < H)
        r3_ok = (r3 >= 0) & (r3 < H)
        c0_ok = (c0 >= 0) & (c0 < W)
        c1_ok = (c1 >= 0) & (c1 < W)
        c2_ok = (c2 >= 0) & (c2 < W)
        c3_ok = (c3 >= 0) & (c3 < W)

        # row offsets (clamped to 0 when invalid; mask out load)
        # combine ic and hw: (IC_BLOCK, BLOCK_HW)
        ic_mask_2d = ic_mask[:, None] & mask_p[None, :]

        def load_xy(r, c, r_ok, c_ok):
            valid = ic_mask_2d & (r_ok & c_ok)[None, :]
            r_safe = tl.where(r_ok, r, 0)
            c_safe = tl.where(c_ok, c, 0)
            off = x_ic_base + r_safe[None, :] * W + c_safe[None, :]
            return tl.load(x_ptr + 0 + off, mask=valid, other=0.0)

        # We'd need x_ptr added; rewrite using x_n_base which already adds x_ptr base via pointer arithmetic
        # Actually x_ic_base = x_n_base + ic*H*W (a pointer). So load from x_ic_base + r*W + c directly.

        def load_p(r, c, r_ok, c_ok):
            valid = ic_mask_2d & (r_ok & c_ok)[None, :]
            r_safe = tl.where(r_ok, r, 0)
            c_safe = tl.where(c_ok, c, 0)
            off = r_safe[None, :] * W + c_safe[None, :]  # (1, BLOCK_HW)
            return tl.load(x_ic_base + off, mask=valid, other=0.0)

        x00 = load_p(r0, c0, r0_ok, c0_ok)
        x01 = load_p(r0, c1, r0_ok, c1_ok)
        x02 = load_p(r0, c2, r0_ok, c2_ok)
        x03 = load_p(r0, c3, r0_ok, c3_ok)

        x10 = load_p(r1, c0, r1_ok, c0_ok)
        x11 = load_p(r1, c1, r1_ok, c1_ok)
        x12 = load_p(r1, c2, r1_ok, c2_ok)
        x13 = load_p(r1, c3, r1_ok, c3_ok)

        x20 = load_p(r2, c0, r2_ok, c0_ok)
        x21 = load_p(r2, c1, r2_ok, c1_ok)
        x22 = load_p(r2, c2, r2_ok, c2_ok)
        x23 = load_p(r2, c3, r2_ok, c3_ok)

        x30 = load_p(r3, c0, r3_ok, c0_ok)
        x31 = load_p(r3, c1, r3_ok, c1_ok)
        x32 = load_p(r3, c2, r3_ok, c2_ok)
        x33 = load_p(r3, c3, r3_ok, c3_ok)

        # Compute conv outputs:
        # out(oh, ow) = sum_{ic, kh, kw} x[ic, oh-1+kh, ow-1+kw] * w[ic, kh, kw]
        # out(h0, w0):
        #   x00*w00 + x01*w01 + x02*w02
        # + x10*w10 + x11*w11 + x12*w12
        # + x20*w20 + x21*w21 + x22*w22
        # out(h0, w0+1):
        #   x01*w00 + x02*w01 + x03*w02
        # + x11*w10 + x12*w11 + x13*w12
        # + x21*w20 + x22*w21 + x23*w22
        # out(h0+1, w0):
        #   x10*w00 + x11*w01 + x12*w02
        # + x20*w10 + x21*w11 + x22*w12
        # + x30*w20 + x31*w21 + x32*w22
        # out(h0+1, w0+1):
        #   x11*w00 + x12*w01 + x13*w02
        # + x21*w10 + x22*w11 + x23*w12
        # + x31*w20 + x32*w21 + x33*w22

        # w?? shape: (IC_BLOCK,). x?? shape: (IC_BLOCK, BLOCK_HW). Multiply -> (IC_BLOCK, BLOCK_HW), sum over IC_BLOCK.
        def acc_out(p00, p01, p02, p10, p11, p12, p20, p21, p22):
            s = (p00 * w00[:, None] + p01 * w01[:, None] + p02 * w02[:, None]
               + p10 * w10[:, None] + p11 * w11[:, None] + p12 * w12[:, None]
               + p20 * w20[:, None] + p21 * w21[:, None] + p22 * w22[:, None])
            return tl.sum(s, axis=0)

        acc00 += acc_out(x00, x01, x02, x10, x11, x12, x20, x21, x22)
        acc01 += acc_out(x01, x02, x03, x11, x12, x13, x21, x22, x23)
        acc10 += acc_out(x10, x11, x12, x20, x21, x22, x30, x31, x32)
        acc11 += acc_out(x11, x12, x13, x21, x22, x23, x31, x32, x33)

    # Add bias
    bias = tl.load(b_ptr + oc)
    acc00 += bias
    acc01 += bias
    acc10 += bias
    acc11 += bias

    # 2x2 max pool
    m = tl.maximum(tl.maximum(acc00, acc01), tl.maximum(acc10, acc11))

    # hardtanh
    m = tl.minimum(tl.maximum(m, hmin), hmax)

    # mask invalid lanes -> 0
    m = tl.where(mask_p, m, 0.0)

    # partial sum across this tile
    partial = tl.sum(m, axis=0)

    # Atomic add to per (n, oc) accumulator
    tl.atomic_add(out_ptr + n * OC + oc, partial)


@triton.jit
def finalize_kernel(
    inp_ptr,   # (B, OC) sums
    out_ptr,   # (B, OC) final tanh(mean)
    N,
    inv_count,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    v = v * inv_count
    e1 = tl.exp(v)
    e2 = tl.exp(-v)
    t = (e1 - e2) / (e1 + e2)
    tl.store(out_ptr + offs, t, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = float(hardtanh_min)
        self.hardtanh_max = float(hardtanh_max)

        # cache for the equivalent conv weight (flipped + transposed)
        self._equiv_weight = None
        self._equiv_weight_version = -1

    def _get_equiv_weight(self):
        # ConvTranspose2d weight shape: (in_channels, out_channels, kH, kW)
        # Equivalent Conv2d weight: (out_channels, in_channels, kH, kW) with kH, kW flipped
        w = self.conv_transpose.weight
        # detect version-id changes
        ver = w._version if hasattr(w, "_version") else 0
        ptr = w.data_ptr()
        key = (ver, ptr)
        if self._equiv_weight is None or getattr(self, "_equiv_weight_key", None) != key:
            with torch.no_grad():
                # flip kH, kW then permute (in_c, out_c, kH, kW) -> (out_c, in_c, kH, kW)
                eq = torch.flip(w, dims=[2, 3]).permute(1, 0, 2, 3).contiguous()
            self._equiv_weight = eq
            self._equiv_weight_key = key
        return self._equiv_weight

    def forward(self, x):
        # Validate configuration matches our specialized kernel
        kH, kW = self.conv_transpose.kernel_size
        sH, sW = (self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride))
        pH, pW = (self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding))

        if not (kH == 3 and kW == 3 and sH == 1 and sW == 1 and pH == 1 and pW == 1
                and self.maxpool_kernel_size == 2 and self.maxpool_stride == 2):
            # Fallback to reference impl
            x = self.conv_transpose(x)
            x = self.maxpool(x)
            x = self.hardtanh(x)
            x = torch.mean(x, dim=(2, 3), keepdim=True)
            x = torch.tanh(x)
            return x

        x = x.contiguous()
        B, IC, H, W = x.shape
        OC = self.out_channels

        if (H % 2 != 0) or (W % 2 != 0):
            x = self.conv_transpose(x)
            x = self.maxpool(x)
            x = self.hardtanh(x)
            x = torch.mean(x, dim=(2, 3), keepdim=True)
            x = torch.tanh(x)
            return x

        pooled_H = H // 2
        pooled_W = W // 2
        inv_count = 1.0 / (pooled_H * pooled_W)

        eq_w = self._get_equiv_weight()  # (OC, IC, 3, 3)
        bias = self.conv_transpose.bias
        if bias is None:
            bias = torch.zeros(OC, device=x.device, dtype=x.dtype)

        sums = torch.zeros((B, OC), device=x.device, dtype=torch.float32)

        BLOCK_HW = 128
        IC_BLOCK = 32

        total_pooled = pooled_H * pooled_W
        num_tiles = (total_pooled + BLOCK_HW - 1) // BLOCK_HW

        grid = (num_tiles, B * OC)

        fused_conv_pool_htanh_mean_tanh_kernel[grid](
            x, eq_w, bias, sums,
            B, IC, H, W,
            OC,
            pooled_H, pooled_W,
            inv_count,
            self.hardtanh_min, self.hardtanh_max,
            BLOCK_HW=BLOCK_HW,
            IC_BLOCK=IC_BLOCK,
            num_warps=4,
            num_stages=2,
        )

        out = torch.empty((B, OC), device=x.device, dtype=x.dtype)
        N = B * OC
        BLOCK_FIN = 256
        grid2 = ((N + BLOCK_FIN - 1) // BLOCK_FIN,)
        finalize_kernel[grid2](sums, out, N, inv_count, BLOCK=BLOCK_FIN, num_warps=2)

        return out.view(B, OC, 1, 1)