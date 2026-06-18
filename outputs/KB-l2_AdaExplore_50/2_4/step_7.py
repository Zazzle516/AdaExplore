import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _mish(x):
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    e = tl.exp(2.0 * sp)
    t = (e - 1.0) / (e + 1.0)
    return x * t


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_im2col_gemm_kernel(
    x_ptr,         # [N, IH, IW, IC] NHWC
    w_ptr,         # [OC, KH*KW*IC] (already reshaped)
    b_ptr,         # [OC]
    y_ptr,         # [N, OH, OW, OC] NHWC
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    M, K_total: tl.constexpr,  # M = N*OH*OW, K = KH*KW*IC
    K, # same as K_total but as runtime param for autotune key
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # decode m -> (n, oh, ow)
    ow = offs_m % OW
    tmp = offs_m // OW
    oh = tmp % OH
    n = tmp // OH

    mask_m = offs_m < M
    mask_n = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K = KH*KW*IC
    offs_k = tl.arange(0, BLOCK_K)
    for k_start in range(0, K_total, BLOCK_K):
        k = k_start + offs_k          # [BLOCK_K]
        mask_k = k < K_total

        # decode k -> (kh, kw, ic)
        ic = k % IC
        tmp_k = k // IC
        kw = tmp_k % KW
        kh = tmp_k // KW

        # input spatial coords (no padding)
        ih = oh[:, None] + kh[None, :]   # [BLOCK_M, BLOCK_K]
        iw = ow[:, None] + kw[None, :]

        # input pointer offset NHWC: ((n*IH + ih)*IW + iw)*IC + ic
        x_off = ((n[:, None] * IH + ih) * IW + iw) * IC + ic[None, :]
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

        # weight: [OC, K] row-major; offset = oc*K + k
        w_off = offs_n[:, None] * K_total + k[None, :]
        w_mask = mask_n[:, None] & mask_k[None, :]
        w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

        # acc += x_vals @ w_vals.T  -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(x_vals, tl.trans(w_vals))

    # bias
    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]

    # double mish
    acc = _mish(acc)
    acc = _mish(acc)

    # store NHWC: y[n, oh, ow, oc]
    y_off = offs_m[:, None] * OC + offs_n[None, :]
    y_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptr + y_off, acc, mask=y_mask)


def conv2d_double_mish(x_nhwc, w_flat, bias, N, IC, IH, IW, OC, OH, OW, KH, KW):
    M = N * OH * OW
    K_total = KH * KW * IC
    y = torch.empty((N, OH, OW, OC), device=x_nhwc.device, dtype=x_nhwc.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OC, meta['BLOCK_N']),
    )
    conv2d_im2col_gemm_kernel[grid](
        x_nhwc, w_flat, bias, y,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        M, K_total, K_total,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # weight: [OC, IC, KH, KW]
        w = conv.weight.data.detach().clone()
        b = conv.bias.data.detach().clone()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # Permute weight to [OC, KH, KW, IC] then flatten last 3 dims
        w_perm = w.permute(0, 2, 3, 1).contiguous()  # [OC, KH, KW, IC]
        OC, KH, KW, IC = w_perm.shape
        w_flat = w_perm.view(OC, KH * KW * IC).contiguous()
        self.register_buffer('w_flat', w_flat)
        self.register_buffer('bias', b.contiguous())
        self.KH = KH
        self.KW = KW

    def forward(self, x):
        x = x.cuda() if not x.is_cuda else x
        # x: [N, IC, IH, IW] -> NHWC
        N, IC, IH, IW = x.shape
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        OC = self.out_channels
        KH = self.KH
        KW = self.KW
        OH = IH - KH + 1
        OW = IW - KW + 1
        y_nhwc = conv2d_double_mish(
            x_nhwc, self.w_flat, self.bias,
            N, IC, IH, IW, OC, OH, OW, KH, KW
        )
        # back to NCHW
        y = y_nhwc.permute(0, 3, 1, 2).contiguous()
        return y