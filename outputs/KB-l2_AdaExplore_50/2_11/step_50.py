import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'IC_TILE': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'IC_TILE': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'IC_TILE': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64, 'IC_TILE': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64, 'IC_TILE': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 128, 'IC_TILE': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 256, 'IC_TILE': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 256, 'IC_TILE': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 256, 'IC_TILE': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 32, 'IC_TILE': 64}, num_warps=4, num_stages=2),
    ],
    key=['IC', 'OC', 'H_out', 'W_out'],
)
@triton.jit
def fused_convt_bn_tanh_kernel(
    x_ptr,           # input: (N, IC, H_in, W_in)
    w_ptr,           # weight: (IC, OC, KH, KW)
    b_ptr,           # bias: (OC,)
    out_ptr,         # output: (N, OC, H_out, W_out)
    bn_scale_ptr,    # (OC,)
    bn_bias_ptr,     # (OC,)
    N, IC, H_in, W_in,
    OC, H_out, W_out,
    KH: tl.constexpr,
    KW: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    IC_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    n = tl.program_id(1)
    oc_block = tl.program_id(2)

    sp_offs = pid * BLOCK_SP + tl.arange(0, BLOCK_SP)  # spatial output positions
    oc_offs = oc_block * BLOCK_OC + tl.arange(0, BLOCK_OC)

    sp_mask = sp_offs < (H_out * W_out)
    oc_mask = oc_offs < OC

    h_out = sp_offs // W_out
    w_out = sp_offs % W_out

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    ic_range = tl.arange(0, IC_TILE)

    for kh in tl.static_range(0, KH):
        h_in = h_out + PAD - kh  # (BLOCK_SP,)
        h_valid = (h_in >= 0) & (h_in < H_in)
        for kw in tl.static_range(0, KW):
            w_in = w_out + PAD - kw  # (BLOCK_SP,)
            w_valid = (w_in >= 0) & (w_in < W_in)
            valid = h_valid & w_valid & sp_mask  # (BLOCK_SP,)

            # Loop over IC in tiles, using tl.dot
            for ic_start in range(0, IC, IC_TILE):
                ic_idx = ic_start + ic_range  # (IC_TILE,)
                ic_mask = ic_idx < IC

                # x[n, ic_idx, h_in[sp], w_in[sp]]: (BLOCK_SP, IC_TILE)
                x_off = (n * IC * H_in * W_in
                         + ic_idx[None, :] * (H_in * W_in)
                         + h_in[:, None] * W_in
                         + w_in[:, None])
                x_m = valid[:, None] & ic_mask[None, :]
                x_tile = tl.load(x_ptr + x_off, mask=x_m, other=0.0)

                # w[ic_idx, oc_offs, kh, kw]: (IC_TILE, BLOCK_OC)
                w_off = (ic_idx[:, None] * (OC * KH * KW)
                         + oc_offs[None, :] * (KH * KW)
                         + kh * KW + kw)
                w_m = ic_mask[:, None] & oc_mask[None, :]
                w_tile = tl.load(w_ptr + w_off, mask=w_m, other=0.0)

                acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    # Apply fused BN: scale * acc + fused_bias (where fused_bias = bn_scale*conv_bias + bn_bias_folded)
    bn_s = tl.load(bn_scale_ptr + oc_offs, mask=oc_mask, other=0.0)
    bn_b = tl.load(bn_bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc * bn_s[None, :] + bn_b[None, :]

    # tanh
    acc = tl.extra.cuda.libdevice.tanh(acc)

    # Store to output: out[n, oc, h_out, w_out]
    # Layout: (N, OC, H_out, W_out)
    out_off = (n * OC * H_out * W_out
               + oc_offs[None, :] * H_out * W_out
               + sp_offs[:, None])
    out_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


@triton.jit
def fused_maxpool_gn_kernel(
    in_ptr,           # (N, C, H, W) - post tanh
    out_ptr,          # (N, C, H_out, W_out)
    gn_weight_ptr,    # (C,)
    gn_bias_ptr,      # (C,)
    N, C, H, W,
    H_out, W_out,
    GROUPS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    SPATIAL_OUT: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS

    offs = tl.arange(0, BLOCK)
    mask = offs < GROUP_SIZE

    c_in_group = offs // SPATIAL_OUT
    sp_out = offs % SPATIAL_OUT
    c = g * CHANNELS_PER_GROUP + c_in_group

    h_out = sp_out // W_out
    w_out = sp_out % W_out
    h_in_base = h_out * 2
    w_in_base = w_out * 2

    base = n * C * H * W + c * H * W
    v00 = tl.load(in_ptr + base + h_in_base * W + w_in_base, mask=mask, other=-1e30)
    v01 = tl.load(in_ptr + base + h_in_base * W + (w_in_base + 1), mask=mask, other=-1e30)
    v10 = tl.load(in_ptr + base + (h_in_base + 1) * W + w_in_base, mask=mask, other=-1e30)
    v11 = tl.load(in_ptr + base + (h_in_base + 1) * W + (w_in_base + 1), mask=mask, other=-1e30)

    m0 = tl.maximum(v00, v01)
    m1 = tl.maximum(v10, v11)
    pooled = tl.maximum(m0, m1)
    pooled = tl.where(mask, pooled, 0.0)

    sum_val = tl.sum(pooled)
    sumsq_val = tl.sum(pooled * pooled)

    mean = sum_val / GROUP_SIZE
    var = sumsq_val / GROUP_SIZE - mean * mean
    inv_std = 1.0 / tl.sqrt(var + EPS)

    gn_w = tl.load(gn_weight_ptr + c, mask=mask, other=0.0)
    gn_b = tl.load(gn_bias_ptr + c, mask=mask, other=0.0)

    normed = (pooled - mean) * inv_std
    result = normed * gn_w + gn_b

    out_base = n * C * SPATIAL_OUT + c * SPATIAL_OUT + sp_out
    tl.store(out_ptr + out_base, result, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_groups = num_groups
        self.gn_eps = 1e-5

    def forward(self, x):
        if self.training or self.stride != 1:
            x = self.conv_transpose(x)
            x = self.batch_norm(x)
            x = torch.tanh(x)
            x = self.max_pool(x)
            x = self.group_norm(x)
            return x

        x = x.contiguous()
        N, IC, H_in, W_in = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        PAD = self.padding
        H_out = H_in + KH - 1 - 2 * PAD
        W_out = W_in + KW - 1 - 2 * PAD

        # BN folded params
        bn = self.batch_norm
        bn_scale = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).contiguous()
        bn_bias_folded = (bn.bias - bn.running_mean * bn_scale)

        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, KH, KW)
        conv_bias = self.conv_transpose.bias if self.conv_transpose.bias is not None else torch.zeros(OC, device=x.device, dtype=x.dtype)
        # Fuse conv bias into bn bias
        bias = (bn_scale * conv_bias + bn_bias_folded).contiguous()

        conv_out = torch.empty((N, OC, H_out, W_out), device=x.device, dtype=x.dtype)

        SP = H_out * W_out
        grid = lambda META: (
            (SP + META['BLOCK_SP'] - 1) // META['BLOCK_SP'],
            N,
            (OC + META['BLOCK_OC'] - 1) // META['BLOCK_OC'],
        )

        fused_convt_bn_tanh_kernel[grid](
            x, weight, bias, conv_out,
            bn_scale, bias,
            N, IC, H_in, W_in,
            OC, H_out, W_out,
            KH=KH, KW=KW, PAD=PAD,
        )

        # Maxpool + GN
        H_pool = H_out // 2
        W_pool = W_out // 2
        out = torch.empty((N, OC, H_pool, W_pool), device=x.device, dtype=x.dtype)

        groups_n = self.num_groups
        channels_per_group = OC // groups_n
        spatial_out = H_pool * W_pool
        group_size = channels_per_group * spatial_out

        BLOCK = 1
        while BLOCK < group_size:
            BLOCK *= 2
        if BLOCK < 256:
            BLOCK = 256

        if BLOCK >= 4096:
            nw = 16
        elif BLOCK >= 2048:
            nw = 8
        else:
            nw = 4

        grid2 = (N * groups_n,)
        fused_maxpool_gn_kernel[grid2](
            conv_out, out,
            self.group_norm.weight.contiguous(),
            self.group_norm.bias.contiguous(),
            N, OC, H_out, W_out,
            H_pool, W_pool,
            GROUPS=groups_n,
            CHANNELS_PER_GROUP=channels_per_group,
            SPATIAL_OUT=spatial_out,
            GROUP_SIZE=group_size,
            BLOCK=BLOCK,
            EPS=self.gn_eps,
            num_warps=nw,
            num_stages=1,
        )

        return out