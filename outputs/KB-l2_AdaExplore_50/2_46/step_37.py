import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_tanh_pool_fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    H_pool, W_pool,
    sub1, sub2,
    stride_xn, stride_xh, stride_xw, stride_xc,  # NHWC
    KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HP: tl.constexpr,
    BLOCK_WP: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_TOTAL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    n_wp_tiles = (W_pool + BLOCK_WP - 1) // BLOCK_WP
    pid_hp = pid_hw // n_wp_tiles
    pid_wp = pid_hw % n_wp_tiles

    BLOCK_HO: tl.constexpr = BLOCK_HP * POOL
    BLOCK_WO: tl.constexpr = BLOCK_WP * POOL
    BLOCK_HW: tl.constexpr = BLOCK_HO * BLOCK_WO

    hp_base = pid_hp * BLOCK_HP
    wp_base = pid_wp * BLOCK_WP

    ho_off = hp_base * POOL + tl.arange(0, BLOCK_HO)
    wo_off = wp_base * POOL + tl.arange(0, BLOCK_WO)

    ho_mask = ho_off < H_out
    wo_mask = wo_off < W_out

    oh = ho_off[:, None] + tl.zeros([1, BLOCK_WO], dtype=tl.int32)
    ow = tl.zeros([BLOCK_HO, 1], dtype=tl.int32) + wo_off[None, :]
    oh_flat = tl.reshape(oh, [BLOCK_HW])
    ow_flat = tl.reshape(ow, [BLOCK_HW])
    hw_mask_2d = ho_mask[:, None] & wo_mask[None, :]
    hw_mask = tl.reshape(hw_mask_2d, [BLOCK_HW])

    oc_off = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_off < C_out

    bias = tl.load(b_ptr + oc_off, mask=oc_mask, other=0.0)

    acc = tl.zeros([BLOCK_HW, BLOCK_OC], dtype=tl.float32)

    for k0 in tl.static_range(0, K_TOTAL, BLOCK_K):
        k_off = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_off < K_TOTAL

        ic = k_off // (KH * KW)
        rem = k_off % (KH * KW)
        kh = rem // KW
        kw = rem % KW

        ih = oh_flat[:, None] + kh[None, :]
        iw = ow_flat[:, None] + kw[None, :]

        in_bounds = hw_mask[:, None] & k_mask[None, :]

        x_offs = (pid_n * stride_xn
                  + ih * stride_xh
                  + iw * stride_xw
                  + ic[None, :] * stride_xc)
        x_vals = tl.load(x_ptr + x_offs, mask=in_bounds, other=0.0)

        w_offs = k_off[:, None] * C_out + oc_off[None, :]
        w_mask = k_mask[:, None] & oc_mask[None, :]
        w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    acc = acc + bias[None, :]
    acc = acc - sub1
    e2x = tl.exp(2.0 * acc)
    t = (e2x - 1.0) / (e2x + 1.0)
    t = t - sub2

    t_3d = tl.reshape(t, [BLOCK_HO, BLOCK_WO, BLOCK_OC])
    t_5d = tl.reshape(t_3d, [BLOCK_HP, POOL, BLOCK_WP, POOL, BLOCK_OC])
    pooled = tl.sum(t_5d, axis=3)
    pooled = tl.sum(pooled, axis=1)
    inv_pool_area = 1.0 / (POOL * POOL)
    pooled = pooled * inv_pool_area

    hp_off = hp_base + tl.arange(0, BLOCK_HP)
    wp_off = wp_base + tl.arange(0, BLOCK_WP)
    hp_mask = hp_off < H_pool
    wp_mask = wp_off < W_pool

    out_offs = (pid_n * (C_out * H_pool * W_pool)
                + oc_off[None, None, :] * (H_pool * W_pool)
                + hp_off[:, None, None] * W_pool
                + wp_off[None, :, None])
    store_mask = hp_mask[:, None, None] & wp_mask[None, :, None] & oc_mask[None, None, :]
    tl.store(out_ptr + out_offs, pooled, mask=store_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = float(subtract1_value)
        self.subtract2_value = float(subtract2_value)
        self.kernel_size_pool = kernel_size_pool
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        w = self.conv.weight.detach()
        OC, C_in, KH, KW = w.shape
        w_perm = w.permute(1, 2, 3, 0).contiguous().view(C_in * KH * KW, OC).contiguous()
        self.register_buffer('w_packed', w_perm.cuda(), persistent=False)
        self.register_buffer('b_packed', self.conv.bias.detach().cuda().contiguous(), persistent=False)

    def forward(self, x):
        x = x.cuda()
        N, C_in, H_in, W_in = x.shape
        C_out = self.out_channels
        KH = KW = self.kernel_size
        H_out = H_in - KH + 1
        W_out = W_in - KW + 1
        POOL = self.kernel_size_pool
        H_pool = H_out // POOL
        W_pool = W_out // POOL

        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        out = torch.empty((N, C_out, H_pool, W_pool), device=x.device, dtype=torch.float32)

        stride_xn = H_in * W_in * C_in
        stride_xh = W_in * C_in
        stride_xw = C_in
        stride_xc = 1

        BLOCK_OC = 128
        BLOCK_HP = 4
        BLOCK_WP = 4
        BLOCK_K = 32
        K_TOTAL = C_in * KH * KW

        n_hp_tiles = (H_pool + BLOCK_HP - 1) // BLOCK_HP
        n_wp_tiles = (W_pool + BLOCK_WP - 1) // BLOCK_WP

        grid = (N, triton.cdiv(C_out, BLOCK_OC), n_hp_tiles * n_wp_tiles)

        conv_tanh_pool_fused_kernel[grid](
            x_nhwc, self.w_packed, self.b_packed, out,
            N, C_in, H_in, W_in,
            C_out, H_out, W_out,
            H_pool, W_pool,
            self.subtract1_value, self.subtract2_value,
            stride_xn, stride_xh, stride_xw, stride_xc,
            KH, KW,
            POOL,
            BLOCK_OC, BLOCK_HP, BLOCK_WP, BLOCK_K,
            K_TOTAL,
            num_warps=8,
            num_stages=3,
        )
        return out