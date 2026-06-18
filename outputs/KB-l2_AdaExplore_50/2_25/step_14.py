import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_min_tanh_gemm_kernel(
    x_ptr,        # NHWC layout: [N, H_IN, W_IN, C_IN]
    w_ptr,        # [K, C_OUT] where K = C_IN*KH*KW
    b_ptr,        # [C_OUT]
    out_ptr,      # [N, H_OUT, W_OUT]
    N, H_IN, W_IN,
    H_OUT, W_OUT,
    KH: tl.constexpr, KW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_sp = tl.program_id(1)

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]
    sp_mask = sp_offs < (H_OUT * W_OUT)
    oh = sp_offs // W_OUT
    ow = sp_offs % W_OUT

    oc_range = tl.arange(0, C_OUT)  # [C_OUT]
    bias = tl.load(b_ptr + oc_range)  # [C_OUT]

    acc = tl.zeros((BLOCK_SP, C_OUT), dtype=tl.float32)

    K = C_IN * KH * KW  # total reduction dimension
    # iterate K dim in tiles of BLOCK_K
    for k_start in tl.static_range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        # decompose k into (ic, kh, kw)
        kw_idx = k_offs % KW
        khic = k_offs // KW
        kh_idx = khic % KH
        ic_idx = khic // KH

        # input gather: x[n, oh+kh, ow+kw, ic] -> NHWC
        # for each (sp, k):
        ih = oh[:, None] + kh_idx[None, :]   # [BLOCK_SP, BLOCK_K]
        iw = ow[:, None] + kw_idx[None, :]
        in_off = ((pid_n * H_IN + ih) * W_IN + iw) * C_IN + ic_idx[None, :]
        x_tile = tl.load(x_ptr + in_off, mask=sp_mask[:, None], other=0.0)  # [BLOCK_SP, BLOCK_K]

        # weight tile: w[k, oc] -> [BLOCK_K, C_OUT]
        w_off = k_offs[:, None] * C_OUT + oc_range[None, :]
        w_tile = tl.load(w_ptr + w_off)  # [BLOCK_K, C_OUT]

        acc += tl.dot(x_tile, w_tile, out_dtype=tl.float32)

    acc = acc + bias[None, :]

    # min across C_OUT
    min_val = tl.min(acc, axis=1)

    t1 = tl.extra.cuda.libdevice.tanh(min_val)
    t2 = tl.extra.cuda.libdevice.tanh(t1)

    out_off = (pid_n * H_OUT + oh) * W_OUT + ow
    tl.store(out_ptr + out_off, t2, mask=sp_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-permute weight: [C_OUT, C_IN, KH, KW] -> [C_IN, KH, KW, C_OUT] -> flat [K, C_OUT]
        w = self.conv.weight.detach()  # [C_OUT, C_IN, KH, KW]
        w_perm = w.permute(1, 2, 3, 0).contiguous()  # [C_IN, KH, KW, C_OUT]
        K = in_channels * kernel_size * kernel_size
        w_flat = w_perm.view(K, out_channels).contiguous()
        self.register_buffer('w_flat', w_flat.cuda())
        self.register_buffer('bias_buf', self.conv.bias.detach().cuda().contiguous())

    def forward(self, x):
        x = x.cuda().contiguous()
        N, C_IN, H_IN, W_IN = x.shape
        # Convert to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [N, H_IN, W_IN, C_IN]

        C_OUT = self.out_channels
        KH = KW = self.kernel_size
        H_OUT = H_IN - KH + 1
        W_OUT = W_IN - KW + 1

        out = torch.empty((N, 1, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

        BLOCK_SP = 64
        BLOCK_K = 16
        grid = (N, triton.cdiv(H_OUT * W_OUT, BLOCK_SP))

        conv_min_tanh_gemm_kernel[grid](
            x_nhwc, self.w_flat, self.bias_buf, out,
            N, H_IN, W_IN,
            H_OUT, W_OUT,
            KH, KW,
            C_IN, C_OUT,
            BLOCK_SP, BLOCK_K,
            num_warps=4,
            num_stages=3,
        )
        return out