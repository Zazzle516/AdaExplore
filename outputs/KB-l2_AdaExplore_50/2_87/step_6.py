import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_nhwc_mish_kernel(
    x_ptr,           # [N, H_IN, W_IN, C_IN] contiguous
    w_ptr,           # [KH, KW, C_IN, C_OUT] contiguous (reordered)
    b_ptr,           # [C_OUT]
    out_ptr,         # [N, H_OUT, W_OUT, C_OUT]
    N, H_IN, W_IN, C_IN,
    C_OUT, H_OUT, W_OUT,
    SUB,
    KH: tl.constexpr, KW: tl.constexpr,
    K_TOTAL: tl.constexpr,  # KH*KW*C_IN, padded for dot
    BLOCK_M: tl.constexpr,   # spatial tile (N*H_OUT*W_OUT axis)
    BLOCK_N: tl.constexpr,   # C_OUT tile
    BLOCK_K: tl.constexpr,   # K reduction tile (must equal K_TOTAL here since small)
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                    # [BLOCK_K]

    # decode (n, oh, ow) from offs_m
    HW_OUT = H_OUT * W_OUT
    n_idx = offs_m // HW_OUT
    hw_idx = offs_m % HW_OUT
    oh = hw_idx // W_OUT
    ow = hw_idx % W_OUT

    mask_m = offs_m < (N * HW_OUT)
    mask_n = offs_n < C_OUT

    # decode k -> (kh, kw, ic)
    # k = (kh * KW + kw) * C_IN + ic
    ic = offs_k % C_IN
    khw = offs_k // C_IN
    kw_i = khw % KW
    kh_i = khw // KW
    mask_k = offs_k < (KH * KW * C_IN)

    # input gather offsets per (m, k):
    # x[n, oh+kh, ow+kw, ic]
    ih = oh[:, None] + kh_i[None, :]   # [BLOCK_M, BLOCK_K]
    iw = ow[:, None] + kw_i[None, :]
    x_off = (n_idx[:, None] * (H_IN * W_IN * C_IN)
             + ih * (W_IN * C_IN)
             + iw * C_IN
             + ic[None, :])
    x_mask = mask_m[:, None] & mask_k[None, :]
    x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

    # weight tile [BLOCK_K, BLOCK_N]
    w_off = offs_k[:, None] * C_OUT + offs_n[None, :]
    w_mask = mask_k[:, None] & mask_n[None, :]
    w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

    acc = tl.dot(x_tile, w_tile, out_dtype=tl.float32)  # [BLOCK_M, BLOCK_N]

    b_val = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b_val[None, :]
    acc = acc - SUB

    # Mish
    sp = tl.where(acc > 20.0, acc, tl.log(1.0 + tl.exp(acc)))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    out = acc * th

    out_off = offs_m[:, None] * C_OUT + offs_n[None, :]
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_off, out, mask=out_mask)


def conv2d_nhwc_mish(x_nhwc, w_kkio, bias, sub, N, H_IN, W_IN, C_IN, C_OUT, KH, KW):
    H_OUT = H_IN - KH + 1
    W_OUT = W_IN - KW + 1

    out = torch.empty((N * H_OUT * W_OUT, C_OUT), device=x_nhwc.device, dtype=torch.float32)

    K_TOTAL = KH * KW * C_IN  # 3*3*8 = 72
    # pad K to power of two for tl.dot
    BLOCK_K = 128 if K_TOTAL <= 128 else 256

    BLOCK_M = 128
    BLOCK_N = 64

    grid = (triton.cdiv(N * H_OUT * W_OUT, BLOCK_M),
            triton.cdiv(C_OUT, BLOCK_N))

    conv2d_nhwc_mish_kernel[grid](
        x_nhwc, w_kkio, bias, out,
        N, H_IN, W_IN, C_IN,
        C_OUT, H_OUT, W_OUT,
        sub,
        KH=KH, KW=KW,
        K_TOTAL=BLOCK_K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    out = out.view(N, H_OUT, W_OUT, C_OUT).permute(0, 3, 1, 2).contiguous()
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.sub_total = float(subtract_value_1 + subtract_value_2)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self._cached_w = None

    def _get_weight_nhwc(self):
        # weight: [C_OUT, C_IN, KH, KW] -> [KH, KW, C_IN, C_OUT]
        w = self.conv.weight.detach()
        w_perm = w.permute(2, 3, 1, 0).contiguous().cuda()
        return w_perm

    def forward(self, x):
        x = x.cuda()
        if not x.is_contiguous():
            x = x.contiguous()
        N, C_IN, H_IN, W_IN = x.shape
        C_OUT = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size

        # NHWC input
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        w_nhwc = self._get_weight_nhwc()
        bias = self.conv.bias.detach().contiguous().cuda()

        return conv2d_nhwc_mish(
            x_nhwc, w_nhwc, bias, self.sub_total,
            N, H_IN, W_IN, C_IN, C_OUT, KH, KW
        )