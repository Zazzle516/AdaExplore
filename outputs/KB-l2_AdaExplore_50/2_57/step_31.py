import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64, 'GROUP_N': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 128, 'GROUP_N': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64, 'GROUP_N': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 128, 'GROUP_N': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 128, 'GROUP_N': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 256, 'GROUP_N': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 128, 'GROUP_N': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 128, 'GROUP_N': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64, 'GROUP_N': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64, 'GROUP_N': 16}, num_warps=4, num_stages=3),
    ],
    key=['N_OUT', 'OC', 'K_TOTAL'],
)
@triton.jit
def conv_relu_hardswish_nhwc_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, IH, IW,
    OC, OH, OW,
    IC: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    K_TOTAL: tl.constexpr,
    N_OUT,
    BLOCK_N: tl.constexpr, BLOCK_OC: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N_OUT, BLOCK_N)
    num_pid_oc = tl.cdiv(OC, BLOCK_OC)

    # L2-friendly swizzle: group rows of N
    num_pid_in_group = GROUP_N * num_pid_oc
    group_id = pid // num_pid_in_group
    first_pid_n = group_id * GROUP_N
    group_size_n = min(num_pid_n - first_pid_n, GROUP_N)
    pid_n = first_pid_n + ((pid % num_pid_in_group) % group_size_n)
    pid_oc = (pid % num_pid_in_group) // group_size_n

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k = tl.arange(0, BLOCK_K)

    mask_n = offs_n < N_OUT
    mask_oc = offs_oc < OC
    mask_k = offs_k < K_TOTAL

    # decompose n -> (b, oh, ow)
    ow = offs_n % OW
    tmp = offs_n // OW
    oh = tmp % OH
    b = tmp // OH

    # decompose k -> (kh, kw, ic): k = kh*KW*IC + kw*IC + ic
    ic_idx = offs_k % IC
    tmp_k = offs_k // IC
    kw_idx = tmp_k % KW
    kh_idx = tmp_k // KW

    # weight layout: (KH*KW*IC, OC) - K-major
    w_offs = offs_k[:, None] * OC + offs_oc[None, :]
    w_mask = mask_k[:, None] & mask_oc[None, :]
    w_tile = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_OC]

    # input layout NHWC: x[b, ih, iw, ic]
    ih = oh[:, None] + kh_idx[None, :]
    iw = ow[:, None] + kw_idx[None, :]
    # stride: b*(IH*IW*IC) + ih*(IW*IC) + iw*IC + ic
    x_offs = (b[:, None] * (IH * IW * IC) +
              ih * (IW * IC) +
              iw * IC +
              ic_idx[None, :])
    x_mask = mask_n[:, None] & mask_k[None, :]
    x_tile = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)  # [BLOCK_N, BLOCK_K]

    acc = tl.dot(x_tile, w_tile)

    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + bias[None, :]

    acc = tl.maximum(acc, 0.0)
    hs = (acc + 3.0) * (1.0 / 6.0)
    hs = tl.minimum(tl.maximum(hs, 0.0), 1.0)
    out = acc * hs

    # store NHWC: y[b, oh, ow, oc] - flat layout: offs_n * OC + offs_oc
    y_offs = offs_n[:, None] * OC + offs_oc[None, :]
    mask = mask_n[:, None] & mask_oc[None, :]
    tl.store(y_ptr + y_offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-permute weight: (OC, IC, KH, KW) -> (KH, KW, IC, OC) flat (KH*KW*IC, OC)
        with torch.no_grad():
            w = self.conv.weight.detach().cuda().contiguous()
            OC, IC, KH, KW = w.shape
            w_perm = w.permute(2, 3, 1, 0).contiguous().view(KH * KW * IC, OC).contiguous()
            self.register_buffer('w_packed', w_perm)
            self.register_buffer('b_packed', self.conv.bias.detach().cuda().contiguous())

    def forward(self, x):
        x = x.cuda()
        B, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1

        # Convert input to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        # NHWC output buffer
        y_nhwc = torch.empty((B, OH, OW, OC), device=x.device, dtype=x.dtype)

        N_OUT = B * OH * OW
        K_TOTAL = IC * KH * KW
        BLOCK_K = 16
        while BLOCK_K < K_TOTAL:
            BLOCK_K *= 2

        grid = lambda meta: (
            triton.cdiv(N_OUT, meta['BLOCK_N']) * triton.cdiv(OC, meta['BLOCK_OC']),
        )

        conv_relu_hardswish_nhwc_kernel[grid](
            x_nhwc, self.w_packed, self.b_packed, y_nhwc,
            B, IH, IW,
            OC, OH, OW,
            IC, KH, KW,
            K_TOTAL,
            N_OUT,
            BLOCK_K=BLOCK_K,
        )
        # NHWC -> NCHW
        return y_nhwc.permute(0, 3, 1, 2).contiguous()