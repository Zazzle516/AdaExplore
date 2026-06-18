import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=8, num_stages=3),
    ],
    key=['IC', 'OC', 'OH', 'OW'],
)
@triton.jit
def fused_conv_sub_hswish_pool_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC,
    OH_pre, OW_pre,
    OH, OW,
    SUB: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)  # post-pool spatial

    oc_mask = oc_offs < OC
    hw_mask = hw_offs < (OH * OW)

    oh = hw_offs // OW
    ow = hw_offs % OW

    # pre-pool top-left positions for each post-pool output
    base_h_pre = oh * POOL
    base_w_pre = ow * POOL

    # We compute conv outputs for POOL*POOL pre-pool positions per post-pool output.
    # Total pre-pool columns per program: POOL*POOL*BLOCK_HW.
    # We'll lay them out as [BLOCK_HW, POOL*POOL] and reduce max along POOL*POOL.

    POOL2: tl.constexpr = POOL * POOL
    NCOLS: tl.constexpr = BLOCK_HW * POOL2

    # Build per-column (within tile) pre-pool h/w positions
    col_arange = tl.arange(0, NCOLS)  # [NCOLS]
    hw_idx = col_arange // POOL2      # which post-pool position (0..BLOCK_HW-1)
    pp_idx = col_arange % POOL2        # which pool offset (0..POOL2-1)
    ph_idx = pp_idx // POOL
    pw_idx = pp_idx % POOL

    # gather base_h_pre[hw_idx] etc.
    # use tl.load equivalent via gather: but we have them as tensors of shape [BLOCK_HW].
    # We can compute via broadcasting: arrange differently.
    # Let's compute h_pre, w_pre for each column directly.
    # h_pre[col] = base_h_pre[hw_idx[col]] + ph_idx[col]
    # We can express via: h_pre = (oh[:, None] * POOL + tl.arange(0,POOL)[None,:]) but POOL2-wise

    # Reshape via broadcast: build [BLOCK_HW, POOL2]
    oh_b = oh[:, None]  # [BLOCK_HW, 1]
    ow_b = ow[:, None]
    hw_mask_b = hw_mask[:, None]

    ph_range = tl.arange(0, POOL2) // POOL  # [POOL2]
    pw_range = tl.arange(0, POOL2) % POOL   # [POOL2]

    h_pre_2d = oh_b * POOL + ph_range[None, :]  # [BLOCK_HW, POOL2]
    w_pre_2d = ow_b * POOL + pw_range[None, :]  # [BLOCK_HW, POOL2]

    valid_2d = hw_mask_b & (h_pre_2d < OH_pre) & (w_pre_2d < OW_pre)  # [BLOCK_HW, POOL2]

    # Flatten to [NCOLS]
    h_pre_flat = tl.reshape(h_pre_2d, (NCOLS,))
    w_pre_flat = tl.reshape(w_pre_2d, (NCOLS,))
    valid_flat = tl.reshape(valid_2d, (NCOLS,))

    # Accumulator: [BLOCK_OC, NCOLS]
    acc = tl.zeros((BLOCK_OC, NCOLS), dtype=tl.float32)

    x_n_base = pid_n * IC * IH * IW

    # Loop over IC, KH, KW
    for ic in range(0, IC):
        x_ic_base = x_n_base + ic * (IH * IW)
        w_ic_base = ic * (KH * KW)
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = h_pre_flat + kh
                iw = w_pre_flat + kw
                in_valid = valid_flat & (ih < IH) & (iw < IW)
                x_off = x_ic_base + ih * IW + iw
                x_v = tl.load(x_ptr + x_off, mask=in_valid, other=0.0)  # [NCOLS]

                w_off = oc_offs * (IC * KH * KW) + w_ic_base + kh * KW + kw
                w_v = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                acc += w_v[:, None] * x_v[None, :]

    # Bias + sub
    b_v = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + b_v[:, None] - SUB

    # HardSwish: x * clamp(x+3, 0, 6) / 6
    xp3 = acc + 3.0
    relu6 = tl.minimum(tl.maximum(xp3, 0.0), 6.0)
    hs = acc * relu6 * (1.0 / 6.0)

    # mask invalid to -inf
    NEG_INF = float('-inf')
    hs = tl.where(valid_flat[None, :], hs, NEG_INF)

    # Reshape to [BLOCK_OC, BLOCK_HW, POOL2] and reduce along last dim
    hs_3d = tl.reshape(hs, (BLOCK_OC, BLOCK_HW, POOL2))
    max_val = tl.max(hs_3d, axis=2)  # [BLOCK_OC, BLOCK_HW]

    # Mish: x * tanh(softplus(x))
    sp = tl.log(1.0 + tl.exp(max_val))
    e2 = tl.exp(2.0 * sp)
    tanh_sp = (e2 - 1.0) / (e2 + 1.0)
    out = max_val * tanh_sp

    out_off = pid_n * OC * OH * OW + oc_offs[:, None] * (OH * OW) + hw_offs[None, :]
    store_mask = oc_mask[:, None] & hw_mask[None, :]
    tl.store(out_ptr + out_off, out, mask=store_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = float(subtract_value)
        self.pool_kernel_size = pool_kernel_size
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC, _, KH, KW = w.shape
        OH_pre = IH - KH + 1
        OW_pre = IW - KW + 1
        POOL = self.pool_kernel_size
        OH = OH_pre // POOL
        OW = OW_pre // POOL

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.float32)

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_OC']),
            triton.cdiv(OH * OW, meta['BLOCK_HW']),
        )

        fused_conv_sub_hswish_pool_mish_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC,
            OH_pre, OW_pre,
            OH, OW,
            self.subtract_value,
            KH, KW, POOL,
        )
        return out