import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K', 'OH', 'OW'],
)
@triton.jit
def conv3x3_implicit_gemm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N_BATCH, IC, IH, IW,
    OC, OH, OW,
    M, N, K: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # row in GEMM = (n, oh, ow)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # col in GEMM = oc
    offs_k = tl.arange(0, K)                          # k = ic * KH * KW + kh*KW + kw

    mask_m = offs_m < M
    mask_n = offs_n < OC

    # decode m -> (n, oh, ow)
    OHW = OH * OW
    n_idx = offs_m // OHW
    rem = offs_m % OHW
    oh = rem // OW
    ow = rem % OW

    # decode k -> (ic, kh, kw)
    KHW = KH * KW
    ic_k = offs_k // KHW
    kk = offs_k % KHW
    kh_k = kk // KW
    kw_k = kk % KW

    # x offsets: (BLOCK_M, K)
    ih = oh[:, None] + kh_k[None, :]
    iw = ow[:, None] + kw_k[None, :]
    x_off = (n_idx[:, None] * (IC * IH * IW)
             + ic_k[None, :] * (IH * IW)
             + ih * IW
             + iw)
    x_mask = mask_m[:, None]
    a = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # (BLOCK_M, K)

    # weight offsets: (K, BLOCK_N) — w layout is (OC, IC, KH, KW)
    w_off = offs_n[None, :] * (IC * KH * KW) + offs_k[:, None]
    w_mask = mask_n[None, :]
    b_w = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # (K, BLOCK_N)

    acc = tl.dot(a, b_w, allow_tf32=True)  # (BLOCK_M, BLOCK_N)

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)
    # HardSwish on relu(x): x * clamp((x+3)/6, 0, 1)
    hs = tl.minimum(tl.maximum((acc + 3.0) / 6.0, 0.0), 1.0)
    acc = acc * hs

    # write back to NCHW: out[n, oc, oh, ow]
    out_off = (n_idx[:, None] * (OC * OHW)
               + offs_n[None, :] * OHW
               + rem[:, None])
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC = w.shape[0]
        KH = w.shape[2]
        KW = w.shape[3]
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        M = N * OH * OW
        K = IC * KH * KW

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),
                             triton.cdiv(OC, meta['BLOCK_N']))

        conv3x3_implicit_gemm_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            M, OC, K,
            KH, KW,
        )
        return out