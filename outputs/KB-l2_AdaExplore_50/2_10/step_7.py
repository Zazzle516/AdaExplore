import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Fused conv (as direct conv with flipped weight, since stride=1, padding=1, kernel=3
# makes ConvTranspose2d equivalent to Conv2d with weight transposed in-out and flipped spatially)
# + 2x2 maxpool + hardtanh + spatial mean + tanh.
# Output: (N, C_out, 1, 1)
#
# Strategy: one program per (n, oc). The program streams over output spatial tiles
# (each tile is BLOCK_TILE pooled outputs = 2*BLOCK_TILE conv outputs in a row),
# computing conv outputs on the fly, taking 2x2 max, applying hardtanh, accumulating
# the sum. After all tiles, divide by total and apply tanh, store.
#
# To make this efficient, we process the output spatially row-by-row in pairs (2 rows
# of conv output -> 1 row of pooled output). For each pair of rows, we tile over
# columns. Inside, we loop over input channels and do the 3x3 conv computation.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_W': 64},  num_warps=4, num_stages=2),
    ],
    key=['IC', 'H', 'W'],
)
@triton.jit
def fused_convT_pool_htanh_rowsum_kernel(
    x_ptr,        # (N, IC, H, W)
    w_ptr,        # (OC, IC, 3, 3) flipped weight
    b_ptr,        # (OC,)
    sum_ptr,      # (N, OC) float32 accumulator (atomic_add)
    N, IC, OC, H, W,
    HTMIN: tl.constexpr,
    HTMAX: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_no = tl.program_id(0)  # n * OC + oc
    ph = tl.program_id(1)      # pooled-row index (0 .. Hp-1)
    n = pid_no // OC
    oc = pid_no % OC

    Wp = W // 2

    bias = tl.load(b_ptr + oc)
    w_oc_base = oc * IC * 9

    h0 = ph * 2
    rm1 = h0 - 1
    r0 = h0
    rp1 = h0 + 1
    rp2 = h0 + 2

    rm1_valid_row = rm1 >= 0
    rp2_valid_row = rp2 < H

    row_acc = tl.zeros([], dtype=tl.float32)

    for pw_start in range(0, Wp, BLOCK_W):
        pw = pw_start + tl.arange(0, BLOCK_W)
        mask_w = pw < Wp
        w0 = pw * 2

        cm1 = w0 - 1
        c0 = w0
        cp1 = w0 + 1
        cp2 = w0 + 2

        cm1_valid = (cm1 >= 0) & mask_w
        c0_valid = (c0 < W) & mask_w
        cp1_valid = (cp1 < W) & mask_w
        cp2_valid = (cp2 < W) & mask_w

        v00 = tl.zeros([BLOCK_W], dtype=tl.float32) + bias
        v01 = tl.zeros([BLOCK_W], dtype=tl.float32) + bias
        v10 = tl.zeros([BLOCK_W], dtype=tl.float32) + bias
        v11 = tl.zeros([BLOCK_W], dtype=tl.float32) + bias

        x_base = n * IC * H * W

        for ic in range(0, IC):
            xb = x_base + ic * H * W
            wb = w_oc_base + ic * 9

            w00 = tl.load(w_ptr + wb + 0)
            w01 = tl.load(w_ptr + wb + 1)
            w02 = tl.load(w_ptr + wb + 2)
            w10 = tl.load(w_ptr + wb + 3)
            w11 = tl.load(w_ptr + wb + 4)
            w12 = tl.load(w_ptr + wb + 5)
            w20 = tl.load(w_ptr + wb + 6)
            w21 = tl.load(w_ptr + wb + 7)
            w22 = tl.load(w_ptr + wb + 8)

            if rm1_valid_row:
                row_off = xb + rm1 * W
                x_rm1_cm1 = tl.load(x_ptr + row_off + cm1, mask=cm1_valid, other=0.0)
                x_rm1_c0  = tl.load(x_ptr + row_off + c0,  mask=c0_valid,  other=0.0)
                x_rm1_cp1 = tl.load(x_ptr + row_off + cp1, mask=cp1_valid, other=0.0)
                x_rm1_cp2 = tl.load(x_ptr + row_off + cp2, mask=cp2_valid, other=0.0)
            else:
                x_rm1_cm1 = tl.zeros([BLOCK_W], dtype=tl.float32)
                x_rm1_c0  = tl.zeros([BLOCK_W], dtype=tl.float32)
                x_rm1_cp1 = tl.zeros([BLOCK_W], dtype=tl.float32)
                x_rm1_cp2 = tl.zeros([BLOCK_W], dtype=tl.float32)

            row_off = xb + r0 * W
            x_r0_cm1 = tl.load(x_ptr + row_off + cm1, mask=cm1_valid, other=0.0)
            x_r0_c0  = tl.load(x_ptr + row_off + c0,  mask=c0_valid,  other=0.0)
            x_r0_cp1 = tl.load(x_ptr + row_off + cp1, mask=cp1_valid, other=0.0)
            x_r0_cp2 = tl.load(x_ptr + row_off + cp2, mask=cp2_valid, other=0.0)

            row_off = xb + rp1 * W
            x_rp1_cm1 = tl.load(x_ptr + row_off + cm1, mask=cm1_valid, other=0.0)
            x_rp1_c0  = tl.load(x_ptr + row_off + c0,  mask=c0_valid,  other=0.0)
            x_rp1_cp1 = tl.load(x_ptr + row_off + cp1, mask=cp1_valid, other=0.0)
            x_rp1_cp2 = tl.load(x_ptr + row_off + cp2, mask=cp2_valid, other=0.0)

            if rp2_valid_row:
                row_off = xb + rp2 * W
                x_rp2_cm1 = tl.load(x_ptr + row_off + cm1, mask=cm1_valid, other=0.0)
                x_rp2_c0  = tl.load(x_ptr + row_off + c0,  mask=c0_valid,  other=0.0)
                x_rp2_cp1 = tl.load(x_ptr + row_off + cp1, mask=cp1_valid, other=0.0)
                x_rp2_cp2 = tl.load(x_ptr + row_off + cp2, mask=cp2_valid, other=0.0)
            else:
                x_rp2_cm1 = tl.zeros([BLOCK_W], dtype=tl.float32)
                x_rp2_c0  = tl.zeros([BLOCK_W], dtype=tl.float32)
                x_rp2_cp1 = tl.zeros([BLOCK_W], dtype=tl.float32)
                x_rp2_cp2 = tl.zeros([BLOCK_W], dtype=tl.float32)

            v00 += x_rm1_cm1 * w00 + x_rm1_c0 * w01 + x_rm1_cp1 * w02
            v00 += x_r0_cm1  * w10 + x_r0_c0  * w11 + x_r0_cp1  * w12
            v00 += x_rp1_cm1 * w20 + x_rp1_c0 * w21 + x_rp1_cp1 * w22

            v01 += x_rm1_c0  * w00 + x_rm1_cp1 * w01 + x_rm1_cp2 * w02
            v01 += x_r0_c0   * w10 + x_r0_cp1  * w11 + x_r0_cp2  * w12
            v01 += x_rp1_c0  * w20 + x_rp1_cp1 * w21 + x_rp1_cp2 * w22

            v10 += x_r0_cm1  * w00 + x_r0_c0  * w01 + x_r0_cp1  * w02
            v10 += x_rp1_cm1 * w10 + x_rp1_c0 * w11 + x_rp1_cp1 * w12
            v10 += x_rp2_cm1 * w20 + x_rp2_c0 * w21 + x_rp2_cp1 * w22

            v11 += x_r0_c0   * w00 + x_r0_cp1  * w01 + x_r0_cp2  * w02
            v11 += x_rp1_c0  * w10 + x_rp1_cp1 * w11 + x_rp1_cp2 * w12
            v11 += x_rp2_c0  * w20 + x_rp2_cp1 * w21 + x_rp2_cp2 * w22

        m = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
        m = tl.minimum(tl.maximum(m, HTMIN), HTMAX)
        m = tl.where(mask_w, m, 0.0)
        row_acc += tl.sum(m, axis=0)

    tl.atomic_add(sum_ptr + pid_no, row_acc)


@triton.jit
def _finalize_tanh_kernel(
    sum_ptr, out_ptr, total_elems, inv_total,
    N_OC,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_OC
    s = tl.load(sum_ptr + offs, mask=mask, other=0.0)
    mean = s * inv_total
    e2x = tl.exp(2.0 * mean)
    out = (e2x - 1.0) / (e2x + 1.0)
    tl.store(out_ptr + offs, out, mask=mask)


def fused_convT_pool_htanh_mean_tanh(x, weight_flipped, bias, hardtanh_min, hardtanh_max):
    N, IC, H, W = x.shape
    OC = weight_flipped.shape[0]
    x = x.contiguous()
    weight_flipped = weight_flipped.contiguous()
    bias = bias.contiguous()

    Hp = H // 2
    Wp = W // 2
    inv_total = 1.0 / (Hp * Wp)

    sum_buf = torch.zeros((N * OC,), device=x.device, dtype=torch.float32)
    out = torch.empty((N, OC, 1, 1), device=x.device, dtype=x.dtype)

    grid = (N * OC, Hp)
    fused_convT_pool_htanh_rowsum_kernel[grid](
        x, weight_flipped, bias, sum_buf,
        N, IC, OC, H, W,
        float(hardtanh_min), float(hardtanh_max),
    )

    N_OC = N * OC
    BLOCK_FIN = 256
    grid_fin = ((N_OC + BLOCK_FIN - 1) // BLOCK_FIN,)
    _finalize_tanh_kernel[grid_fin](
        sum_buf, out, Hp * Wp, float(inv_total),
        N_OC,
        BLOCK=BLOCK_FIN,
        num_warps=4,
    )
    return out


# Fallback fused pool+htanh+mean+tanh kernel (used if shapes mismatch assumptions)
@triton.jit
def fused_pool_htanh_mean_tanh_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    H_out, W_out,
    HTMIN: tl.constexpr,
    HTMAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    base = n * C * H * W + c * H * W
    total = H_out * W_out
    inv = 1.0 / total

    acc = tl.zeros([], dtype=tl.float32)

    for start in range(0, total, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < total
        ho = offs // W_out
        wo = offs % W_out
        h0 = ho * 2
        w0 = wo * 2

        i00 = base + h0 * W + w0
        i01 = base + h0 * W + (w0 + 1)
        i10 = base + (h0 + 1) * W + w0
        i11 = base + (h0 + 1) * W + (w0 + 1)

        v00 = tl.load(x_ptr + i00, mask=mask, other=-1e30)
        v01 = tl.load(x_ptr + i01, mask=mask, other=-1e30)
        v10 = tl.load(x_ptr + i10, mask=mask, other=-1e30)
        v11 = tl.load(x_ptr + i11, mask=mask, other=-1e30)

        m = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
        m = tl.minimum(tl.maximum(m, HTMIN), HTMAX)
        m = tl.where(mask, m, 0.0)
        acc += tl.sum(m, axis=0)

    mean_val = acc * inv
    e2x = tl.exp(2.0 * mean_val)
    out_val = (e2x - 1.0) / (e2x + 1.0)
    tl.store(out_ptr + pid, out_val)


def fused_pool_htanh_mean_tanh(x, hardtanh_min, hardtanh_max):
    N, C, H, W = x.shape
    H_out = H // 2
    W_out = W // 2
    x = x.contiguous()
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    grid = (N * C,)
    fused_pool_htanh_mean_tanh_kernel[grid](
        x, out,
        N, C, H, W,
        H_out, W_out,
        float(hardtanh_min), float(hardtanh_max),
        BLOCK=1024,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Cache for the transformed weight
        self._cached_weight = None
        self._cached_weight_version = None

    def _get_flipped_weight(self):
        # ConvTranspose2d weight shape: (in_channels, out_channels, kH, kW)
        # Equivalent Conv2d weight: (out_channels, in_channels, kH, kW) with kernel flipped spatially
        w = self.conv_transpose.weight
        # transpose first two dims, flip last two dims
        w_eq = w.transpose(0, 1).flip(dims=(2, 3)).contiguous()
        return w_eq

    def forward(self, x):
        # Check if we can use the fused fast path: stride=1, padding=1, kernel=3, maxpool 2x2 stride 2
        can_fast = (
            self.kernel_size == 3 and self.stride == 1 and self.padding == 1 and
            self.maxpool_kernel_size == 2 and self.maxpool_stride == 2 and
            x.shape[2] % 2 == 0 and x.shape[3] % 2 == 0
        )

        if can_fast:
            x = x.contiguous()
            w_eq = self._get_flipped_weight()
            b = self.conv_transpose.bias
            if b is None:
                b = torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)
            return fused_convT_pool_htanh_mean_tanh(
                x, w_eq, b, self.hardtanh_min, self.hardtanh_max
            )
        else:
            x = self.conv_transpose(x)
            if (self.maxpool_kernel_size == 2 and self.maxpool_stride == 2 and
                    x.shape[2] % 2 == 0 and x.shape[3] % 2 == 0):
                return fused_pool_htanh_mean_tanh(x, self.hardtanh_min, self.hardtanh_max)
            x = F.max_pool2d(x, self.maxpool_kernel_size, self.maxpool_stride)
            x = F.hardtanh(x, self.hardtanh_min, self.hardtanh_max)
            x = torch.mean(x, dim=(2, 3), keepdim=True)
            x = torch.tanh(x)
            return x