import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_P': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_P': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_P': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_P': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_P': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_P': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_P': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_P': 4}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'PH', 'PW', 'IC', 'KH', 'KW', 'POOL'],
)
@triton.jit
def fused_conv_tanh_scale_bias_pool_kernel(
    x_ptr, w_ptr, cb_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    PH, PW,
    scale,
    KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    IC_C: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_p = tl.program_id(2)

    num_pools = PH * PW
    PP: tl.constexpr = POOL * POOL
    TILE: tl.constexpr = BLOCK_P * PP
    K: tl.constexpr = IC_C * KH * KW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    flat = tl.arange(0, TILE)
    p_local = flat // PP
    ij = flat % PP
    ii = ij // POOL
    jj = ij % POOL

    p_idx = pid_p * BLOCK_P + p_local
    p_mask = p_idx < num_pools

    ph = p_idx // PW
    pw = p_idx % PW
    oh = ph * POOL + ii
    ow = pw * POOL + jj

    cb = tl.load(cb_ptr + oc_offs, mask=oc_mask, other=0.0)
    bb = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    acc = tl.zeros((BLOCK_OC, TILE), dtype=tl.float32)

    base_x = pid_n * IC * IH * IW

    # Build weight tile [BLOCK_OC, K] using NHWC weight layout: [OC, KH, KW, IC]
    # K dim ordering: kh * KW * IC + kw * IC + ic  (matches NHWC gather order)
    k_idx = tl.arange(0, K)
    w_offs = oc_offs[:, None] * K + k_idx[None, :]
    w_mask = oc_mask[:, None] & (k_idx[None, :] < K)
    w_tile = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)  # [BLOCK_OC, K]

    # Build x tile [K, TILE] via im2col gather using NHWC input layout: [N, IH, IW, IC]
    # K decomposes as (kh, kw, ic)
    kh_idx = k_idx // (KW * IC_C)
    rem = k_idx % (KW * IC_C)
    kw_idx = rem // IC_C
    ic_idx = rem % IC_C

    # x is NHWC [N, IH, IW, IC]
    # offset = base + (oh+kh)*IW*IC + (ow+kw)*IC + ic
    ih_t = oh[None, :] + kh_idx[:, None]   # [K, TILE]
    iw_t = ow[None, :] + kw_idx[:, None]   # [K, TILE]
    x_off = base_x + ih_t * (IW * IC_C) + iw_t * IC_C + ic_idx[:, None]
    x_mask = p_mask[None, :] & (k_idx[:, None] < K)
    x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [K, TILE]

    acc = tl.dot(w_tile, x_tile, acc, allow_tf32=True)

    acc = acc + cb[:, None]
    t = 2.0 * acc
    tanh_val = 2.0 / (1.0 + tl.exp(-t)) - 1.0
    val = tanh_val * scale + bb[:, None]

    neg_inf = float('-inf')
    val = tl.where(p_mask[None, :], val, neg_inf)

    val = tl.reshape(val, (BLOCK_OC, BLOCK_P, PP))
    max_acc = tl.max(val, axis=2)

    p_out = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    p_out_mask = p_out < num_pools
    out_off = (pid_n * OC + oc_offs[:, None]) * (PH * PW) + p_out[None, :]
    mask = oc_mask[:, None] & p_out_mask[None, :]
    tl.store(out_ptr + out_off, max_acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scaling_factor = float(scaling_factor)
        self.pool_kernel_size = pool_kernel_size

        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # Store weight as NHWC: [OC, KH, KW, IC]
        w = conv.weight.detach().clone()  # [OC, IC, KH, KW]
        w_nhwc = w.permute(0, 2, 3, 1).contiguous()
        self.weight_nhwc = nn.Parameter(w_nhwc)
        self.conv_bias = nn.Parameter(conv.bias.detach().clone())
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, IH, IW = x.shape
        KH = KW = self.kernel_size
        OC = self.out_channels
        OH = IH - KH + 1
        OW = IW - KW + 1
        POOL = self.pool_kernel_size
        PH = OH // POOL
        PW = OW // POOL

        # Convert input to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        out = torch.empty((N, OC, PH, PW), device=x.device, dtype=x.dtype)

        bias_flat = self.bias.view(-1).contiguous()
        weight = self.weight_nhwc.contiguous()
        conv_bias = self.conv_bias.contiguous()

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_OC']),
            triton.cdiv(PH * PW, meta['BLOCK_P']),
        )

        fused_conv_tanh_scale_bias_pool_kernel[grid](
            x_nhwc, weight, conv_bias, bias_flat, out,
            N, IC, IH, IW,
            OC, OH, OW,
            PH, PW,
            self.scaling_factor,
            KH, KW,
            POOL,
            IC,
        )

        return out