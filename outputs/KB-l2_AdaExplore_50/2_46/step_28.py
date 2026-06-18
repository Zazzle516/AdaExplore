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
    stride_wk, stride_woc,                         # W: [K, OC], K = C_in*KH*KW
    KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    C_IN: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    pool_total = H_pool * W_pool
    hw_off = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_off < pool_total

    ph = hw_off // W_pool
    pw = hw_off % W_pool

    oc_off = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_off < C_out

    # bias
    bias = tl.load(b_ptr + oc_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

    # Final pool accumulator: [BLOCK_HW, BLOCK_OC]
    pool_acc = tl.zeros([BLOCK_HW, BLOCK_OC], dtype=tl.float32)
    inv_pool_area = 1.0 / (POOL * POOL)

    K_total = C_IN * KH * KW

    # For each conv-output in pool window
    for dh in tl.static_range(POOL):
        for dw in tl.static_range(POOL):
            oh = ph * POOL + dh  # [BLOCK_HW]
            ow = pw * POOL + dw

            # GEMM: acc[BLOCK_HW, BLOCK_OC] = sum_k x[BLOCK_HW, k] * w[k, BLOCK_OC]
            acc = tl.zeros([BLOCK_HW, BLOCK_OC], dtype=tl.float32)

            for k0 in range(0, K_total, BLOCK_K):
                k_off = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
                k_mask = k_off < K_total

                # Decompose k -> (ic, kh, kw)
                ic = k_off // (KH * KW)
                rem = k_off % (KH * KW)
                kh = rem // KW
                kw = rem % KW

                ih = oh[:, None] + kh[None, :]  # [BLOCK_HW, BLOCK_K]
                iw = ow[:, None] + kw[None, :]

                in_bounds = (ih < H_in) & (iw < W_in) & hw_mask[:, None] & k_mask[None, :]

                x_offs = (pid_n * stride_xn
                          + ih * stride_xh
                          + iw * stride_xw
                          + ic[None, :] * stride_xc)
                x_vals = tl.load(x_ptr + x_offs, mask=in_bounds, other=0.0)  # [BLOCK_HW, BLOCK_K]

                w_offs = k_off[:, None] * stride_wk + oc_off[None, :] * stride_woc
                w_mask = k_mask[:, None] & oc_mask[None, :]
                w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_OC]

                acc += tl.dot(x_vals, w_vals)

            acc = acc + bias[None, :]
            acc = acc - sub1
            # tanh
            e2x = tl.exp(2.0 * acc)
            t = (e2x - 1.0) / (e2x + 1.0)
            t = t - sub2
            pool_acc += t

    pool_acc = pool_acc * inv_pool_area

    # Store NCHW output: out[n, oc, ph, pw]
    out_offs = (pid_n * (C_out * H_pool * W_pool)
                + oc_off[None, :] * (H_pool * W_pool)
                + ph[:, None] * W_pool
                + pw[:, None])
    store_mask = hw_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_offs, pool_acc, mask=store_mask)


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

        # Precompute weight in [K, OC] layout: K = C_in * KH * KW
        w = self.conv.weight.detach()  # [OC, C_in, KH, KW]
        OC, C_in, KH, KW = w.shape
        # Permute to [C_in, KH, KW, OC] -> reshape to [K, OC]
        w_perm = w.permute(1, 2, 3, 0).contiguous().view(C_in * KH * KW, OC)
        self.register_buffer('w_packed', w_perm.cuda(), persistent=False)
        self.register_buffer('b_packed', self.conv.bias.detach().cuda(), persistent=False)

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

        # Convert input to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        out = torch.empty((N, C_out, H_pool, W_pool), device=x.device, dtype=torch.float32)

        # Strides for NHWC layout
        stride_xn = H_in * W_in * C_in
        stride_xh = W_in * C_in
        stride_xw = C_in
        stride_xc = 1

        stride_wk = C_out  # row stride (K dim)
        stride_woc = 1     # col stride (OC dim)

        BLOCK_OC = 64
        BLOCK_HW = 64
        BLOCK_K = 32

        grid = (N, triton.cdiv(C_out, BLOCK_OC), triton.cdiv(H_pool * W_pool, BLOCK_HW))

        conv_tanh_pool_fused_kernel[grid](
            x_nhwc, self.w_packed, self.b_packed, out,
            N, C_in, H_in, W_in,
            C_out, H_out, W_out,
            H_pool, W_pool,
            self.subtract1_value, self.subtract2_value,
            stride_xn, stride_xh, stride_xw, stride_xc,
            stride_wk, stride_woc,
            KH, KW,
            POOL,
            BLOCK_OC, BLOCK_HW, BLOCK_K,
            C_in,
            num_warps=4,
            num_stages=2,
        )
        return out