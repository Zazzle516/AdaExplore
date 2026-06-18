import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose_bn_tanh_maxpool_kernel(
    x_ptr,           # input: (N, IC, H_in, W_in)
    w_ptr,           # weight: (IC, OC, KH, KW)
    conv_bias_ptr,   # (OC,)
    bn_scale_ptr,    # (OC,)
    bn_bias_ptr,     # (OC,)
    out_ptr,         # output: (N, OC, H_out, W_out)
    N, IC, H_in, W_in,
    OC, H_pre, W_pre,
    H_out, W_out,
    KH: tl.constexpr,
    KW: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    # one program per (n, oc_tile, out_spatial_tile_after_pool)
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_oc = tl.program_id(2)

    n = pid_n
    oc_start = pid_oc * BLOCK_OC
    sp_start = pid * BLOCK_SP

    offs_oc = oc_start + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    offs_sp = sp_start + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]

    mask_oc = offs_oc < OC
    mask_sp = offs_sp < (H_out * W_out)

    # decompose output (pooled) spatial coord
    h_out = offs_sp // W_out
    w_out = offs_sp % W_out

    # For each of the 4 pooled-input positions, compute conv_transpose output
    # h_pre = 2*h_out + dh, w_pre = 2*w_out + dw, dh, dw in {0,1}
    # Conv transpose with stride=1, padding=PAD:
    # out[h_pre, w_pre] = sum_{ic, kh, kw} x[ic, h_pre + PAD - kh, w_pre + PAD - kw] * w[ic, oc, kh, kw]
    # valid when 0 <= h_pre + PAD - kh < H_in

    # Accumulators for 4 sub-positions
    acc00 = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # Loop over kh, kw, ic
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            # h_in = h_pre + PAD - kh, for h_pre = 2*h_out, 2*h_out+1
            for dh in tl.static_range(0, 2):
                for dw in tl.static_range(0, 2):
                    h_pre = h_out * 2 + dh
                    w_pre = w_out * 2 + dw
                    h_in = h_pre + PAD - kh
                    w_in = w_pre + PAD - kw

                    valid_h = (h_in >= 0) & (h_in < H_in)
                    valid_w = (w_in >= 0) & (w_in < W_in)
                    valid = valid_h & valid_w & mask_sp  # [BLOCK_SP]

                    # gather over IC: accumulate using matmul-like dot
                    # We need: sum_ic x[n, ic, h_in, w_in] * w[ic, oc, kh, kw]
                    # x shape: load [BLOCK_SP, IC]
                    # w shape: load [IC, BLOCK_OC]

                    # but IC may be large; loop over IC in chunks too
                    BLOCK_IC: tl.constexpr = 16
                    n_ic_tiles = (IC + BLOCK_IC - 1) // BLOCK_IC

                    acc_local = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)
                    for ic_t in range(n_ic_tiles):
                        offs_ic = ic_t * BLOCK_IC + tl.arange(0, BLOCK_IC)
                        mask_ic = offs_ic < IC

                        # x[n, ic, h_in, w_in] -> ptr: n*IC*H_in*W_in + ic*H_in*W_in + h_in*W_in + w_in
                        x_offsets = (n * IC * H_in * W_in
                                     + offs_ic[None, :] * H_in * W_in
                                     + h_in[:, None] * W_in
                                     + w_in[:, None])
                        x_mask = valid[:, None] & mask_ic[None, :]
                        x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

                        # w[ic, oc, kh, kw] -> ptr: ic*OC*KH*KW + oc*KH*KW + kh*KW + kw
                        w_offsets = (offs_ic[:, None] * OC * KH * KW
                                     + offs_oc[None, :] * KH * KW
                                     + kh * KW + kw)
                        w_mask = mask_ic[:, None] & mask_oc[None, :]
                        w_vals = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)

                        acc_local += tl.dot(x_vals, w_vals)

                    if (dh == 0) & (dw == 0):
                        acc00 += acc_local
                    if (dh == 0) & (dw == 1):
                        acc01 += acc_local
                    if (dh == 1) & (dw == 0):
                        acc10 += acc_local
                    if (dh == 1) & (dw == 1):
                        acc11 += acc_local

    # Add conv bias, BN, tanh
    conv_b = tl.load(conv_bias_ptr + offs_oc, mask=mask_oc, other=0.0)  # [BLOCK_OC]
    bn_s = tl.load(bn_scale_ptr + offs_oc, mask=mask_oc, other=0.0)
    bn_b = tl.load(bn_bias_ptr + offs_oc, mask=mask_oc, other=0.0)

    # add bias
    acc00 = acc00 + conv_b[None, :]
    acc01 = acc01 + conv_b[None, :]
    acc10 = acc10 + conv_b[None, :]
    acc11 = acc11 + conv_b[None, :]

    # BN: y = x * bn_s + bn_b
    acc00 = acc00 * bn_s[None, :] + bn_b[None, :]
    acc01 = acc01 * bn_s[None, :] + bn_b[None, :]
    acc10 = acc10 * bn_s[None, :] + bn_b[None, :]
    acc11 = acc11 * bn_s[None, :] + bn_b[None, :]

    # tanh
    t00 = tl.extra.cuda.libdevice.tanh(acc00)
    t01 = tl.extra.cuda.libdevice.tanh(acc01)
    t10 = tl.extra.cuda.libdevice.tanh(acc10)
    t11 = tl.extra.cuda.libdevice.tanh(acc11)

    # max pool 2x2
    m0 = tl.maximum(t00, t01)
    m1 = tl.maximum(t10, t11)
    pooled = tl.maximum(m0, m1)  # [BLOCK_SP, BLOCK_OC]

    # store: out[n, oc, h_out, w_out]
    out_offsets = (n * OC * H_out * W_out
                   + offs_oc[None, :] * H_out * W_out
                   + offs_sp[:, None])
    out_mask = mask_sp[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_offsets, pooled, mask=out_mask)


@triton.jit
def group_norm_kernel(
    inout_ptr,       # (N, C, S)
    gn_weight_ptr,   # (C,)
    gn_bias_ptr,     # (C,)
    N, C, S,
    GROUPS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS

    offs = tl.arange(0, BLOCK)
    num_chunks = (GROUP_SIZE + BLOCK - 1) // BLOCK

    sum_val = 0.0
    sumsq_val = 0.0

    for chunk in range(num_chunks):
        idx = chunk * BLOCK + offs
        mask = idx < GROUP_SIZE

        c_in_group = idx // S
        sp = idx % S
        c = g * CHANNELS_PER_GROUP + c_in_group

        base = n * C * S + c * S + sp
        v = tl.load(inout_ptr + base, mask=mask, other=0.0)
        v_masked = tl.where(mask, v, 0.0)
        sum_val += tl.sum(v_masked)
        sumsq_val += tl.sum(v_masked * v_masked)

    mean = sum_val / GROUP_SIZE
    var = sumsq_val / GROUP_SIZE - mean * mean
    inv_std = 1.0 / tl.sqrt(var + EPS)

    for chunk in range(num_chunks):
        idx = chunk * BLOCK + offs
        mask = idx < GROUP_SIZE

        c_in_group = idx // S
        sp = idx % S
        c = g * CHANNELS_PER_GROUP + c_in_group

        gn_w = tl.load(gn_weight_ptr + c, mask=mask, other=0.0)
        gn_b = tl.load(gn_bias_ptr + c, mask=mask, other=0.0)

        base = n * C * S + c * S + sp
        v = tl.load(inout_ptr + base, mask=mask, other=0.0)
        normed = (v - mean) * inv_std
        result = normed * gn_w + gn_b
        tl.store(inout_ptr + base, result, mask=mask)


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
        if self.training:
            x = self.conv_transpose(x)
            x = self.batch_norm(x)
            x = self.tanh(x)
            x = self.max_pool(x)
            x = self.group_norm(x)
            return x

        N, IC, H_in, W_in = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        PAD = self.padding

        # output of conv_transpose: H_pre = (H_in - 1)*stride - 2*pad + kh
        H_pre = (H_in - 1) * self.stride - 2 * PAD + KH
        W_pre = (W_in - 1) * self.stride - 2 * PAD + KW

        H_out = H_pre // 2
        W_out = W_pre // 2

        # BN folded
        bn = self.batch_norm
        bn_scale = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).contiguous()
        bn_bias = (bn.bias - bn.running_mean * bn_scale).contiguous()

        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()  # (IC, OC, KH, KW)
        conv_bias = self.conv_transpose.bias.contiguous() if self.conv_transpose.bias is not None else torch.zeros(OC, device=x.device, dtype=x.dtype)

        out = torch.empty((N, OC, H_out, W_out), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32 if OC <= 64 else 64
        BLOCK_SP = 64

        sp_total = H_out * W_out
        n_sp_tiles = (sp_total + BLOCK_SP - 1) // BLOCK_SP
        n_oc_tiles = (OC + BLOCK_OC - 1) // BLOCK_OC

        grid = (n_sp_tiles, N, n_oc_tiles)

        conv_transpose_bn_tanh_maxpool_kernel[grid](
            x, w, conv_bias, bn_scale, bn_bias, out,
            N, IC, H_in, W_in,
            OC, H_pre, W_pre,
            H_out, W_out,
            KH=KH, KW=KW, PAD=PAD,
            BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
            num_warps=4, num_stages=2,
        )

        # Group norm
        S = H_out * W_out
        groups = self.num_groups
        channels_per_group = OC // groups
        group_size = channels_per_group * S

        if group_size <= 256:
            BLOCK = 256
        elif group_size <= 512:
            BLOCK = 512
        elif group_size <= 1024:
            BLOCK = 1024
        else:
            BLOCK = 1024

        grid_gn = (N * groups,)
        group_norm_kernel[grid_gn](
            out,
            self.group_norm.weight.contiguous(),
            self.group_norm.bias.contiguous(),
            N, OC, S,
            GROUPS=groups,
            CHANNELS_PER_GROUP=channels_per_group,
            GROUP_SIZE=group_size,
            BLOCK=BLOCK,
            EPS=self.gn_eps,
            num_warps=4,
        )

        return out