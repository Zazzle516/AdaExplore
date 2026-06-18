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
    BLOCK_HW: tl.constexpr,
    IC_BLOCK: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid % OC

    total = pooled_H * pooled_W
    w_oc_base = w_ptr + oc * IC * 9
    x_n_base = x_ptr + n * IC * H * W
    bias = tl.load(b_ptr + oc)

    sum_acc = tl.zeros((1,), dtype=tl.float32)

    for tile_id in range(0, NUM_TILES):
        offs = tile_id * BLOCK_HW + tl.arange(0, BLOCK_HW)
        mask_p = offs < total
        ph = offs // pooled_W
        pw = offs - ph * pooled_W
        h0 = ph * 2
        w0 = pw * 2

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

        r0_s = tl.where(r0_ok, r0, 0)
        r1_s = tl.where(r1_ok, r1, 0)
        r2_s = tl.where(r2_ok, r2, 0)
        r3_s = tl.where(r3_ok, r3, 0)
        c0_s = tl.where(c0_ok, c0, 0)
        c1_s = tl.where(c1_ok, c1, 0)
        c2_s = tl.where(c2_ok, c2, 0)
        c3_s = tl.where(c3_ok, c3, 0)

        acc00 = tl.zeros((BLOCK_HW,), dtype=tl.float32)
        acc01 = tl.zeros((BLOCK_HW,), dtype=tl.float32)
        acc10 = tl.zeros((BLOCK_HW,), dtype=tl.float32)
        acc11 = tl.zeros((BLOCK_HW,), dtype=tl.float32)

        for ic_start in range(0, IC, IC_BLOCK):
            ic_offs = ic_start + tl.arange(0, IC_BLOCK)
            ic_mask = ic_offs < IC

            w_base = w_oc_base + ic_offs * 9
            w00 = tl.load(w_base + 0, mask=ic_mask, other=0.0)
            w01 = tl.load(w_base + 1, mask=ic_mask, other=0.0)
            w02 = tl.load(w_base + 2, mask=ic_mask, other=0.0)
            w10 = tl.load(w_base + 3, mask=ic_mask, other=0.0)
            w11 = tl.load(w_base + 4, mask=ic_mask, other=0.0)
            w12 = tl.load(w_base + 5, mask=ic_mask, other=0.0)
            w20 = tl.load(w_base + 6, mask=ic_mask, other=0.0)
            w21 = tl.load(w_base + 7, mask=ic_mask, other=0.0)
            w22 = tl.load(w_base + 8, mask=ic_mask, other=0.0)

            x_ic_base = x_n_base + ic_offs[:, None] * (H * W)
            ic_mask_2d = ic_mask[:, None] & mask_p[None, :]

            m_r0c0 = ic_mask_2d & (r0_ok & c0_ok)[None, :]
            m_r0c1 = ic_mask_2d & (r0_ok & c1_ok)[None, :]
            m_r0c2 = ic_mask_2d & (r0_ok & c2_ok)[None, :]
            m_r0c3 = ic_mask_2d & (r0_ok & c3_ok)[None, :]
            m_r1c0 = ic_mask_2d & (r1_ok & c0_ok)[None, :]
            m_r1c1 = ic_mask_2d & (r1_ok & c1_ok)[None, :]
            m_r1c2 = ic_mask_2d & (r1_ok & c2_ok)[None, :]
            m_r1c3 = ic_mask_2d & (r1_ok & c3_ok)[None, :]
            m_r2c0 = ic_mask_2d & (r2_ok & c0_ok)[None, :]
            m_r2c1 = ic_mask_2d & (r2_ok & c1_ok)[None, :]
            m_r2c2 = ic_mask_2d & (r2_ok & c2_ok)[None, :]
            m_r2c3 = ic_mask_2d & (r2_ok & c3_ok)[None, :]
            m_r3c0 = ic_mask_2d & (r3_ok & c0_ok)[None, :]
            m_r3c1 = ic_mask_2d & (r3_ok & c1_ok)[None, :]
            m_r3c2 = ic_mask_2d & (r3_ok & c2_ok)[None, :]
            m_r3c3 = ic_mask_2d & (r3_ok & c3_ok)[None, :]

            v00 = tl.load(x_ic_base + r0_s[None, :] * W + c0_s[None, :], mask=m_r0c0, other=0.0)
            v01 = tl.load(x_ic_base + r0_s[None, :] * W + c1_s[None, :], mask=m_r0c1, other=0.0)
            v02 = tl.load(x_ic_base + r0_s[None, :] * W + c2_s[None, :], mask=m_r0c2, other=0.0)
            v03 = tl.load(x_ic_base + r0_s[None, :] * W + c3_s[None, :], mask=m_r0c3, other=0.0)

            v10 = tl.load(x_ic_base + r1_s[None, :] * W + c0_s[None, :], mask=m_r1c0, other=0.0)
            v11 = tl.load(x_ic_base + r1_s[None, :] * W + c1_s[None, :], mask=m_r1c1, other=0.0)
            v12 = tl.load(x_ic_base + r1_s[None, :] * W + c2_s[None, :], mask=m_r1c2, other=0.0)
            v13 = tl.load(x_ic_base + r1_s[None, :] * W + c3_s[None, :], mask=m_r1c3, other=0.0)

            v20 = tl.load(x_ic_base + r2_s[None, :] * W + c0_s[None, :], mask=m_r2c0, other=0.0)
            v21 = tl.load(x_ic_base + r2_s[None, :] * W + c1_s[None, :], mask=m_r2c1, other=0.0)
            v22 = tl.load(x_ic_base + r2_s[None, :] * W + c2_s[None, :], mask=m_r2c2, other=0.0)
            v23 = tl.load(x_ic_base + r2_s[None, :] * W + c3_s[None, :], mask=m_r2c3, other=0.0)

            v30 = tl.load(x_ic_base + r3_s[None, :] * W + c0_s[None, :], mask=m_r3c0, other=0.0)
            v31 = tl.load(x_ic_base + r3_s[None, :] * W + c1_s[None, :], mask=m_r3c1, other=0.0)
            v32 = tl.load(x_ic_base + r3_s[None, :] * W + c2_s[None, :], mask=m_r3c2, other=0.0)
            v33 = tl.load(x_ic_base + r3_s[None, :] * W + c3_s[None, :], mask=m_r3c3, other=0.0)

            s00 = (v00 * w00[:, None] + v01 * w01[:, None] + v02 * w02[:, None]
                 + v10 * w10[:, None] + v11 * w11[:, None] + v12 * w12[:, None]
                 + v20 * w20[:, None] + v21 * w21[:, None] + v22 * w22[:, None])
            s01 = (v01 * w00[:, None] + v02 * w01[:, None] + v03 * w02[:, None]
                 + v11 * w10[:, None] + v12 * w11[:, None] + v13 * w12[:, None]
                 + v21 * w20[:, None] + v22 * w21[:, None] + v23 * w22[:, None])
            s10 = (v10 * w00[:, None] + v11 * w01[:, None] + v12 * w02[:, None]
                 + v20 * w10[:, None] + v21 * w11[:, None] + v22 * w12[:, None]
                 + v30 * w20[:, None] + v31 * w21[:, None] + v32 * w22[:, None])
            s11 = (v11 * w00[:, None] + v12 * w01[:, None] + v13 * w02[:, None]
                 + v21 * w10[:, None] + v22 * w11[:, None] + v23 * w12[:, None]
                 + v31 * w20[:, None] + v32 * w21[:, None] + v33 * w22[:, None])

            acc00 += tl.sum(s00, axis=0)
            acc01 += tl.sum(s01, axis=0)
            acc10 += tl.sum(s10, axis=0)
            acc11 += tl.sum(s11, axis=0)

        acc00 += bias
        acc01 += bias
        acc10 += bias
        acc11 += bias

        m = tl.maximum(tl.maximum(acc00, acc01), tl.maximum(acc10, acc11))
        m = tl.minimum(tl.maximum(m, hmin), hmax)
        m = tl.where(mask_p, m, 0.0)
        sum_acc += tl.sum(m, axis=0, keep_dims=True)

    mean = tl.sum(sum_acc, axis=0) * inv_count
    e1 = tl.exp(mean)
    e2 = tl.exp(-mean)
    t = (e1 - e2) / (e1 + e2)
    tl.store(out_ptr + n * OC + oc, t)


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

        BLOCK_HW = 128
        IC_BLOCK = 32

        total_pooled = pooled_H * pooled_W
        num_tiles = (total_pooled + BLOCK_HW - 1) // BLOCK_HW

        out = torch.empty((B, OC), device=x.device, dtype=x.dtype)

        grid = (B * OC,)

        fused_conv_pool_htanh_mean_tanh_kernel[grid](
            x, eq_w, bias, out,
            B, IC, H, W,
            OC,
            pooled_H, pooled_W,
            inv_count,
            self.hardtanh_min, self.hardtanh_max,
            BLOCK_HW=BLOCK_HW,
            IC_BLOCK=IC_BLOCK,
            NUM_TILES=num_tiles,
            num_warps=4,
            num_stages=2,
        )

        return out.view(B, OC, 1, 1)