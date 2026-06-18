import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 256}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'OC', 'H_OUT', 'W_OUT', 'KH', 'KW'],
)
@triton.jit
def conv2d_div_lrelu_gemm_kernel(
    x_ptr,        # input NHWC: [N, H_IN, W_IN, IC]
    w_ptr,        # weight packed [OC, K_PAD]
    b_ptr,        # bias [OC]
    out_ptr,      # output NCHW: [N, OC, H_OUT, W_OUT]
    N, IC: tl.constexpr, H_IN, W_IN,
    OC, H_OUT, W_OUT,
    KH: tl.constexpr, KW: tl.constexpr,
    K_PAD: tl.constexpr,   # padded K (IC*KH*KW rounded up)
    K_REAL: tl.constexpr,  # actual K = IC*KH*KW
    inv_divisor, neg_slope,
    BLOCK_OC: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_k = tl.arange(0, K_PAD)

    HW = H_OUT * W_OUT
    mask_oc = offs_oc < OC
    mask_hw = offs_hw < HW
    mask_k = offs_k < K_REAL

    out_h = offs_hw // W_OUT  # [BLOCK_HW]
    out_w = offs_hw % W_OUT

    # Decompose k into (ic, kh, kw) where layout is kh * KW * IC + kw * IC + ic
    # (matches NHWC channels-last input ordering)
    kh_idx = offs_k // (KW * IC)
    rem = offs_k % (KW * IC)
    kw_idx = rem // IC
    ic_idx = rem % IC

    # Input pointer (NHWC): x[n, h, w, ic]
    # offset = n * (H_IN*W_IN*IC) + h*W_IN*IC + w*IC + ic
    in_h = out_h[:, None] + kh_idx[None, :]  # [BLOCK_HW, K_PAD]
    in_w = out_w[:, None] + kw_idx[None, :]
    x_off = (pid_n * H_IN * W_IN * IC
             + in_h * W_IN * IC
             + in_w * IC
             + ic_idx[None, :])

    x_mask = mask_hw[:, None] & mask_k[None, :]
    x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [BLOCK_HW, K_PAD]

    # Weight: [OC, K_PAD] row-major
    w_off = offs_oc[:, None] * K_PAD + offs_k[None, :]
    w_mask = mask_oc[:, None] & mask_k[None, :]
    w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # [BLOCK_OC, K_PAD]

    # acc[BLOCK_OC, BLOCK_HW] = w_tile @ x_tile.T
    acc = tl.dot(w_tile, tl.trans(x_tile))

    b_vals = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + b_vals[:, None]
    acc = acc * inv_divisor
    acc = tl.where(acc >= 0, acc, acc * neg_slope)

    out_offset = (pid_n * OC * HW
                  + offs_oc[:, None] * HW
                  + offs_hw[None, :])
    mask = mask_oc[:, None] & mask_hw[None, :]
    tl.store(out_ptr + out_offset, acc, mask=mask)


def _next_pow2(x):
    p = 1
    while p < x:
        p <<= 1
    return p


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        KH = kernel_size
        KW = kernel_size
        IC = in_channels
        OC = out_channels
        K_REAL = IC * KH * KW
        K_PAD = max(16, _next_pow2(K_REAL))

        # Pre-pack weight: [OC, IC, KH, KW] -> [OC, KH, KW, IC] flat -> pad to K_PAD
        w = self.conv.weight.detach().clone()  # [OC, IC, KH, KW]
        # permute to [OC, KH, KW, IC]
        w_perm = w.permute(0, 2, 3, 1).contiguous().view(OC, K_REAL)
        if K_PAD > K_REAL:
            w_padded = torch.zeros((OC, K_PAD), dtype=w_perm.dtype)
            w_padded[:, :K_REAL] = w_perm
        else:
            w_padded = w_perm
        self.register_buffer('weight_packed', w_padded.cuda())
        self.register_buffer('bias_packed', self.conv.bias.detach().clone().cuda())

        self.K_REAL = K_REAL
        self.K_PAD = K_PAD

    def forward(self, x):
        x = x.cuda()
        N, IC, H_IN, W_IN = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        H_OUT = H_IN - KH + 1
        W_OUT = W_IN - KW + 1

        # Convert to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        out = torch.empty((N, OC, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

        inv_divisor = 1.0 / float(self.divisor)
        neg_slope = 0.01

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_OC']),
            triton.cdiv(H_OUT * W_OUT, meta['BLOCK_HW']),
        )

        conv2d_div_lrelu_gemm_kernel[grid](
            x_nhwc, self.weight_packed, self.bias_packed, out,
            N, IC, H_IN, W_IN,
            OC, H_OUT, W_OUT,
            KH, KW,
            self.K_PAD, self.K_REAL,
            inv_divisor, neg_slope,
        )
        return out