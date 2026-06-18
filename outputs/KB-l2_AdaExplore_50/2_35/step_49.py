import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 16, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 16, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 32, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 16, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 16, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OUT_HW', 'IC'],
)
@triton.jit
def fused_conv_hs_pool_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, OH, OW,
    POH, POW,
    OUT_HW,
    SUBV,
    KH: tl.constexpr,
    KW: tl.constexpr,
    POOL_K: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    POOL_SQ: tl.constexpr = POOL_K * POOL_K
    SP_PRE: tl.constexpr = BLOCK_SP * POOL_SQ

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    oc_mask = oc_offs < OC
    sp_mask = sp_offs < OUT_HW

    ph = sp_offs // POW  # [BLOCK_SP]
    pw = sp_offs % POW

    pool_idx = tl.arange(0, POOL_SQ)
    pi_arr = pool_idx // POOL_K
    pj_arr = pool_idx % POOL_K

    oh_2d = ph[:, None] * POOL_K + pi_arr[None, :]  # [BLOCK_SP, POOL_SQ]
    ow_2d = pw[:, None] * POOL_K + pj_arr[None, :]
    valid_2d = sp_mask[:, None] & (oh_2d < OH) & (ow_2d < OW)

    oh_flat = tl.reshape(oh_2d, (SP_PRE,))
    ow_flat = tl.reshape(ow_2d, (SP_PRE,))
    valid_flat = tl.reshape(valid_2d, (SP_PRE,))

    acc = tl.zeros((BLOCK_OC, SP_PRE), dtype=tl.float32)

    WIC = W * IC
    HWIC = H * W * IC
    KWIC = KW * IC

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh_flat + kh
            iw = ow_flat + kw
            for ic_start in range(0, IC, BLOCK_K):
                ic_offs = ic_start + tl.arange(0, BLOCK_K)
                ic_mask = ic_offs < IC

                # w[oc, kh, kw, ic] layout [OC, KH, KW, IC]
                w_ptrs = w_ptr + (
                    oc_offs[:, None] * (KH * KW * IC)
                    + kh * KWIC + kw * IC
                    + ic_offs[None, :]
                )
                w_vals = tl.load(
                    w_ptrs,
                    mask=oc_mask[:, None] & ic_mask[None, :],
                    other=0.0,
                )

                # x[n, ih, iw, ic] layout [N, H, W, IC]
                x_ptrs = x_ptr + (
                    pid_n * HWIC
                    + ih[:, None] * WIC
                    + iw[:, None] * IC
                    + ic_offs[None, :]
                )
                x_vals = tl.load(
                    x_ptrs,
                    mask=valid_flat[:, None] & ic_mask[None, :],
                    other=0.0,
                )

                acc += tl.dot(w_vals, tl.trans(x_vals))

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[:, None] - SUBV

    shifted = tl.minimum(tl.maximum(acc + 3.0, 0.0), 6.0)
    hs = acc * shifted * (1.0 / 6.0)

    NEG_INF = float('-inf')
    hs = tl.where(valid_flat[None, :], hs, NEG_INF)

    hs_3d = tl.reshape(hs, (BLOCK_OC, BLOCK_SP, POOL_SQ))
    pooled = tl.max(hs_3d, axis=2)

    # mish: x * tanh(softplus(x))
    sp_val = tl.log(1.0 + tl.exp(pooled))
    e2 = tl.exp(2.0 * sp_val)
    tanh_sp = 1.0 - 2.0 / (e2 + 1.0)
    out = pooled * tanh_sp

    out_ptrs = out_ptr + (
        pid_n * (OC * OUT_HW)
        + oc_offs[:, None] * OUT_HW
        + sp_offs[None, :]
    )
    tl.store(out_ptrs, out, mask=oc_mask[:, None] & sp_mask[None, :])


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = float(subtract_value)
        self.pool_kernel_size = int(pool_kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.cuda()
        w = self.conv.weight.cuda()
        b = self.conv.bias.cuda().contiguous()

        N, IC, H, W = x.shape
        OC, _, KH, KW = w.shape
        OH = H - KH + 1
        OW = W - KW + 1
        POOL_K = self.pool_kernel_size
        POH = OH // POOL_K
        POW = OW // POOL_K
        OUT_HW = POH * POW

        # NHWC layout: [N, H, W, IC]
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        # weight layout: [OC, KH, KW, IC]
        w_ohwi = w.permute(0, 2, 3, 1).contiguous()

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=torch.float32)

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_OC']),
            triton.cdiv(OUT_HW, meta['BLOCK_SP']),
        )

        fused_conv_hs_pool_mish_kernel[grid](
            x_nhwc, w_ohwi, b, out,
            N, IC, H, W,
            OC, OH, OW,
            POH, POW,
            OUT_HW,
            self.subtract_value,
            KH, KW,
            POOL_K,
        )

        return out