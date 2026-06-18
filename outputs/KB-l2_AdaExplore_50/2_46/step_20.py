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
    stride_xn, stride_xh, stride_xw, stride_xc,  # x is NHWC
    stride_wk, stride_wc,                          # w is (KH*KW*C_in, C_out)
    stride_on, stride_oc, stride_oh, stride_ow,   # out is NCHW
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
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = offs_hw < pool_total

    ph = offs_hw // W_pool
    pw = offs_hw % W_pool

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = offs_oc < C_out

    bias = tl.load(b_ptr + offs_oc, mask=oc_mask, other=0.0)

    inv_pool_area = 1.0 / (POOL * POOL)

    # accumulator for pooled output [BLOCK_HW, BLOCK_OC]
    pool_acc = tl.zeros([BLOCK_HW, BLOCK_OC], dtype=tl.float32)

    K_total = C_IN * KH * KW

    for dh in tl.static_range(POOL):
        for dw in tl.static_range(POOL):
            oh = ph * POOL + dh  # [BLOCK_HW]
            ow = pw * POOL + dw

            # GEMM accumulator [BLOCK_HW, BLOCK_OC]
            conv_acc = tl.zeros([BLOCK_HW, BLOCK_OC], dtype=tl.float32)

            # K loop over (kh, kw, ic)
            for k_start in range(0, K_total, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)
                k_mask = offs_k < K_total

                # decompose k -> kh, kw, ic
                kh = offs_k // (KW * C_IN)
                kw_ic = offs_k % (KW * C_IN)
                kw = kw_ic // C_IN
                ic = kw_ic % C_IN

                # x indices: [BLOCK_HW, BLOCK_K]
                ih = oh[:, None] + kh[None, :]  # [BLOCK_HW, BLOCK_K]
                iw = ow[:, None] + kw[None, :]
                ic_b = ic[None, :]  # [1, BLOCK_K]

                x_off = (pid_n * stride_xn
                         + ih * stride_xh
                         + iw * stride_xw
                         + ic_b * stride_xc)
                x_load_mask = hw_mask[:, None] & k_mask[None, :]
                x_vals = tl.load(x_ptr + x_off, mask=x_load_mask, other=0.0)

                # w indices: [BLOCK_K, BLOCK_OC]
                w_off = offs_k[:, None] * stride_wk + offs_oc[None, :] * stride_wc
                w_load_mask = k_mask[:, None] & oc_mask[None, :]
                w_vals = tl.load(w_ptr + w_off, mask=w_load_mask, other=0.0)

                conv_acc += tl.dot(x_vals, w_vals, allow_tf32=True)

            conv_acc = conv_acc + bias[None, :]
            conv_acc = conv_acc - sub1
            # tanh via stable formula
            e2x = tl.exp(2.0 * conv_acc)
            t = (e2x - 1.0) / (e2x + 1.0)
            t = t - sub2
            pool_acc += t

    pool_acc = pool_acc * inv_pool_area

    # store: out[pid_n, offs_oc, ph, pw]
    out_off = (pid_n * stride_on
               + offs_oc[None, :] * stride_oc
               + ph[:, None] * stride_oh
               + pw[:, None] * stride_ow)
    store_mask = hw_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, pool_acc, mask=store_mask)


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

        # Pre-transform weight to (KH*KW*C_in, C_out)
        with torch.no_grad():
            w = self.conv.weight.detach().clone().cuda()  # [C_out, C_in, KH, KW]
            # We want layout: K = (kh, kw, ic), so permute to [KH, KW, C_in, C_out]
            w_t = w.permute(2, 3, 1, 0).contiguous()
            w_flat = w_t.view(kernel_size * kernel_size * in_channels, out_channels).contiguous()
        self.register_buffer("w_packed", w_flat, persistent=False)
        self.register_buffer("b_packed", self.conv.bias.detach().clone().cuda().contiguous(), persistent=False)

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

        # Convert x to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        out = torch.empty((N, C_out, H_pool, W_pool), device=x.device, dtype=torch.float32)

        BLOCK_OC = 64
        BLOCK_HW = 32
        BLOCK_K = 32

        grid = (N, triton.cdiv(C_out, BLOCK_OC), triton.cdiv(H_pool * W_pool, BLOCK_HW))

        conv_tanh_pool_fused_kernel[grid](
            x_nhwc, self.w_packed, self.b_packed, out,
            N, C_in, H_in, W_in,
            C_out, H_out, W_out,
            H_pool, W_pool,
            self.subtract1_value, self.subtract2_value,
            x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
            self.w_packed.stride(0), self.w_packed.stride(1),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            KH, KW,
            POOL,
            BLOCK_OC, BLOCK_HW, BLOCK_K,
            C_in,
            num_warps=4,
            num_stages=2,
        )
        return out