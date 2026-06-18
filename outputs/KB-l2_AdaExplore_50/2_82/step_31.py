import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_PH': 2, 'BLOCK_PW': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_PH': 4, 'BLOCK_PW': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_PH': 4, 'BLOCK_PW': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PH': 2, 'BLOCK_PW': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PH': 2, 'BLOCK_PW': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PH': 4, 'BLOCK_PW': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_PH': 2, 'BLOCK_PW': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_PH': 4, 'BLOCK_PW': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_PH': 2, 'BLOCK_PW': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_PH': 4, 'BLOCK_PW': 4}, num_warps=4, num_stages=3),
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
    BLOCK_PH: tl.constexpr,
    BLOCK_PW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_p = tl.program_id(2)

    num_pw_blocks = (PW + BLOCK_PW - 1) // BLOCK_PW
    num_ph_blocks = (PH + BLOCK_PH - 1) // BLOCK_PH
    pid_ph = pid_p // num_pw_blocks
    pid_pw = pid_p % num_pw_blocks

    PP: tl.constexpr = POOL * POOL
    BLOCK_P: tl.constexpr = BLOCK_PH * BLOCK_PW
    TILE: tl.constexpr = BLOCK_P * PP

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Compute (ph, pw) for each pool position in the block; (ii, jj) inside pool window.
    flat = tl.arange(0, TILE)
    p_local = flat // PP
    ij = flat % PP
    ii = ij // POOL
    jj = ij % POOL

    ph_local = p_local // BLOCK_PW
    pw_local = p_local % BLOCK_PW

    ph = pid_ph * BLOCK_PH + ph_local
    pw = pid_pw * BLOCK_PW + pw_local

    p_mask = (ph < PH) & (pw < PW)

    oh = ph * POOL + ii
    ow = pw * POOL + jj

    cb = tl.load(cb_ptr + oc_offs, mask=oc_mask, other=0.0)
    bb = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    acc = tl.zeros((BLOCK_OC, TILE), dtype=tl.float32)

    base_x = pid_n * IC * IH * IW
    base_offs = oh * IW + ow

    for ic in range(0, IC):
        ic_x_base = base_x + ic * IH * IW
        ic_w_base = ic * (KH * KW)
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                in_off = ic_x_base + base_offs + kh * IW + kw
                x_val = tl.load(x_ptr + in_off, mask=p_mask, other=0.0)

                w_off = oc_offs * (IC * KH * KW) + ic_w_base + kh * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)

                acc += w_val[:, None] * x_val[None, :]

    acc = acc + cb[:, None]
    t = 2.0 * acc
    tanh_val = 2.0 / (1.0 + tl.exp(-t)) - 1.0
    val = tanh_val * scale + bb[:, None]

    neg_inf = float('-inf')
    val = tl.where(p_mask[None, :], val, neg_inf)

    val = tl.reshape(val, (BLOCK_OC, BLOCK_P, PP))
    max_acc = tl.max(val, axis=2)  # [BLOCK_OC, BLOCK_P]

    # Store
    p_local_out = tl.arange(0, BLOCK_P)
    ph_out = pid_ph * BLOCK_PH + (p_local_out // BLOCK_PW)
    pw_out = pid_pw * BLOCK_PW + (p_local_out % BLOCK_PW)
    p_out_mask = (ph_out < PH) & (pw_out < PW)
    p_out_idx = ph_out * PW + pw_out

    out_off = (pid_n * OC + oc_offs[:, None]) * (PH * PW) + p_out_idx[None, :]
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
            triton.cdiv(PH, meta['BLOCK_PH']) * triton.cdiv(PW, meta['BLOCK_PW']),
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