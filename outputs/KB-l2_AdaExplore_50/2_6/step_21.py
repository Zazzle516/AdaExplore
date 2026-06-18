import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OW': 8}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_OW': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OW': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_OW': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OW': 16}, num_warps=8, num_stages=2),
    ],
    key=['N', 'OC', 'IC', 'D', 'H', 'W', 'OD', 'OH', 'OW', 'KD', 'KH', 'KW', 'POOL'],
)
@triton.jit
def fused_conv_softmax_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, D, H, W,
    OC, KD, KH, KW,
    CD, CH, CW,        # conv output dims
    OD, OH, OW,        # pooled output dims
    POOL: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ow_blocks = (OW + BLOCK_OW - 1) // BLOCK_OW
    owb = pid % num_ow_blocks
    tmp = pid // num_ow_blocks
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    n = tmp // OD

    ow_offs = owb * BLOCK_OW + tl.arange(0, BLOCK_OW)
    ow_mask = ow_offs < OW

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC
    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < (IC * KD * KH * KW)

    # decode k -> (ic, kd, kh, kw)
    KDHW = KD * KH * KW
    KHW = KH * KW
    ic_idx = k_offs // KDHW
    rem = k_offs % KDHW
    kd_idx = rem // KHW
    rem2 = rem % KHW
    kh_idx = rem2 // KW
    kw_idx = rem2 % KW

    # Load weight: (OC, IC*KD*KH*KW) = (BLOCK_OC, BLOCK_K)
    w_ptrs = w_ptr + oc_offs[:, None] * (IC * KDHW) + k_offs[None, :]
    w_mask = oc_mask[:, None] & k_mask[None, :]
    w = tl.load(w_ptrs, mask=w_mask, other=0.0)

    # Load bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    # Pool window in conv-output space
    cd0 = od * POOL
    ch0 = oh * POOL
    cw0_base = ow_offs * POOL  # [BLOCK_OW]

    DHW = D * H * W
    HW = H * W

    # Final pooled max for softmax(conv) over BLOCK_OC x BLOCK_OW
    pool_max = tl.zeros([BLOCK_OC, BLOCK_OW], dtype=tl.float32) - float('inf')

    for ddp in tl.static_range(0, POOL):
        cd = cd0 + ddp
        for dhp in tl.static_range(0, POOL):
            ch = ch0 + dhp
            for dwp in tl.static_range(0, POOL):
                cw = cw0_base + dwp  # [BLOCK_OW]

                # Compute conv output for (n, :, cd, ch, cw) over OC channels
                # x patch: gather x[n, ic, cd+kd, ch+kh, cw+kw] for each k
                # x_index = n*IC*DHW + ic*DHW + (cd+kd)*HW + (ch+kh)*W + (cw+kw)
                d_in = cd + kd_idx  # [BLOCK_K]
                h_in = ch + kh_idx
                w_in_base = kw_idx   # [BLOCK_K]

                # x indices: shape [BLOCK_OW, BLOCK_K]
                x_idx = (n * IC * DHW
                         + ic_idx[None, :] * DHW
                         + d_in[None, :] * HW
                         + h_in[None, :] * W
                         + cw[:, None] + w_in_base[None, :])
                x_mask = ow_mask[:, None] & k_mask[None, :]
                xv = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)  # [BLOCK_OW, BLOCK_K]

                # GEMM: conv_out[oc, ow] = sum_k w[oc,k] * x[ow,k]
                # => [BLOCK_OC, BLOCK_OW] = w [BLOCK_OC, BLOCK_K] @ xv.T [BLOCK_K, BLOCK_OW]
                conv_out = tl.dot(w, tl.trans(xv))  # [BLOCK_OC, BLOCK_OW]
                conv_out = conv_out + bias[:, None]
                conv_out = tl.where(oc_mask[:, None] & ow_mask[None, :], conv_out, -float('inf'))

                # softmax along OC axis
                m = tl.max(conv_out, axis=0)  # [BLOCK_OW]
                ex = tl.exp(conv_out - m[None, :])
                ex = tl.where(oc_mask[:, None] & ow_mask[None, :], ex, 0.0)
                s = tl.sum(ex, axis=0)  # [BLOCK_OW]
                sm = ex / s[None, :]
                sm = tl.where(oc_mask[:, None] & ow_mask[None, :], sm, -float('inf'))

                pool_max = tl.maximum(pool_max, sm)

    # Store
    ODOHOW = OD * OH * OW
    out_base = n * OC * ODOHOW + od * OH * OW + oh * OW
    out_ptrs = out_ptr + out_base + ow_offs[None, :] + oc_offs[:, None] * ODOHOW
    out_mask = oc_mask[:, None] & ow_mask[None, :]
    tl.store(out_ptrs, pool_max, mask=out_mask)


def fused_conv_softmax_pool(x, weight, bias, pool_total):
    N, IC, D, H, W = x.shape
    OC, _, KD, KH, KW = weight.shape
    CD = D - KD + 1
    CH = H - KH + 1
    CW = W - KW + 1
    OD = CD // pool_total
    OH = CH // pool_total
    OW = CW // pool_total
    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = triton.next_power_of_2(OC)
    if BLOCK_OC < 16:
        BLOCK_OC = 16
    K_total = IC * KD * KH * KW
    BLOCK_K = triton.next_power_of_2(K_total)
    if BLOCK_K < 16:
        BLOCK_K = 16

    w_flat = weight.reshape(OC, K_total).contiguous()

    grid = lambda meta: (N * OD * OH * ((OW + meta['BLOCK_OW'] - 1) // meta['BLOCK_OW']),)
    fused_conv_softmax_pool_kernel[grid](
        x, w_flat, bias, out,
        N, IC, D, H, W,
        OC, KD, KH, KW,
        CD, CH, CW,
        OD, OH, OW,
        POOL=pool_total,
        BLOCK_OC=BLOCK_OC,
        BLOCK_K=BLOCK_K,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self.pool_total = pool_kernel_size * pool_kernel_size

    def forward(self, x):
        x = x.contiguous()
        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()
        out = fused_conv_softmax_pool(x, weight, bias, self.pool_total)
        return out