import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_P': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_P': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_P': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_P': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_P': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_P': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_P': 16}, num_warps=4, num_stages=2),
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
    BLOCK_OC: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # program ids: (n, oc_block, pool_block)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_p = tl.program_id(2)

    num_pools = PH * PW
    PP: tl.constexpr = POOL * POOL
    TILE: tl.constexpr = BLOCK_P * PP

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # flat indices over BLOCK_P pool positions x PP conv positions per pool
    flat = tl.arange(0, TILE)
    p_local = flat // PP        # which pool position within block
    ij = flat % PP              # which conv position within the pool window
    ii = ij // POOL
    jj = ij % POOL

    p_idx = pid_p * BLOCK_P + p_local
    p_mask = p_idx < num_pools

    ph = p_idx // PW
    pw = p_idx % PW
    oh = ph * POOL + ii         # [TILE]
    ow = pw * POOL + jj         # [TILE]

    # load conv bias + bias for each oc
    cb = tl.load(cb_ptr + oc_offs, mask=oc_mask, other=0.0)
    bb = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    # accumulator: [BLOCK_OC, TILE]
    acc = tl.zeros((BLOCK_OC, TILE), dtype=tl.float32)

    base_x = pid_n * IC * IH * IW
    for ic in range(0, IC):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = oh + kh
                iw = ow + kw
                in_off = base_x + ic * IH * IW + ih * IW + iw
                x_val = tl.load(x_ptr + in_off, mask=p_mask, other=0.0)  # [TILE]

                w_off = oc_offs * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                acc += w_val[:, None] * x_val[None, :]

    # apply conv bias, tanh, scale, bias
    acc = acc + cb[:, None]
    t = 2.0 * acc
    tanh_val = 2.0 / (1.0 + tl.exp(-t)) - 1.0
    val = tanh_val * scale + bb[:, None]  # [BLOCK_OC, TILE]

    # mask invalid pool positions to -inf so they don't affect max
    neg_inf = float('-inf')
    val = tl.where(p_mask[None, :], val, neg_inf)

    # reshape [BLOCK_OC, BLOCK_P, PP] and max over PP
    val = tl.reshape(val, (BLOCK_OC, BLOCK_P, PP))
    max_acc = tl.max(val, axis=2)  # [BLOCK_OC, BLOCK_P]

    # store output
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
        self.weight = nn.Parameter(conv.weight.detach().clone())
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

        out = torch.empty((N, OC, PH, PW), device=x.device, dtype=x.dtype)

        bias_flat = self.bias.view(-1).contiguous()
        weight = self.weight.contiguous()
        conv_bias = self.conv_bias.contiguous()

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_OC']),
            triton.cdiv(PH * PW, meta['BLOCK_P']),
        )

        fused_conv_tanh_scale_bias_pool_kernel[grid](
            x, weight, conv_bias, bias_flat, out,
            N, IC, IH, IW,
            OC, OH, OW,
            PH, PW,
            self.scaling_factor,
            KH, KW,
            POOL,
        )

        return out