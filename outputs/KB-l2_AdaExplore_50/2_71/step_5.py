import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K', 'OH', 'OW'],
)
@triton.jit
def conv2d_im2col_kernel(
    x_ptr,        # NCHW: [B, IC, H, W]
    w_ptr,        # [KH, KW, IC, OC]
    b_ptr,        # [OC]
    out_ptr,      # NCHW: [B, OC, OH, OW]
    B, H, W, IC,
    OC, KH, KW,
    OH, OW,
    M, N, K,      # M = B*OH*OW, N = OC, K = KH*KW*IC
    inv_div,
    neg_slope,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IC_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # decode m -> (b, oh, ow)
    ow = offs_m % OW
    tmp = offs_m // OW
    oh = tmp % OH
    b = tmp // OH

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K = KH*KW*IC; iterate in chunks of BLOCK_K
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]
        mask_k = k_idx < K

        # decode k -> (kh, kw, ic)
        ic = k_idx % IC_C
        tmp2 = k_idx // IC_C
        kw = tmp2 % KW
        kh = tmp2 // KW

        # x: NCHW offset = ((b*IC + ic)*H + ih)*W + iw
        ih = oh[:, None] + kh[None, :]   # [M, K]
        iw = ow[:, None] + kw[None, :]   # [M, K]
        b_b = b[:, None]                 # [M, 1]
        ic_b = ic[None, :]               # [1, K]

        x_offset = ((b_b * IC_C + ic_b) * H + ih) * W + iw
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_tile = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # w: [KH, KW, IC, OC] offset = ((kh*KW + kw)*IC + ic)*OC + oc
        w_offset = ((kh[:, None] * KW + kw[:, None]) * IC_C + ic[:, None]) * OC + offs_n[None, :]
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]
    acc = acc * inv_div
    acc = tl.where(acc >= 0, acc, acc * neg_slope)

    # store NCHW: out[b, oc, oh, ow]
    out_offset = (b[:, None] * OC + offs_n[None, :]) * (OH * OW) + (oh[:, None] * OW + ow[:, None])
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offset, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = float(divisor)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # pre-permute weight to [KH, KW, IC, OC]
        with torch.no_grad():
            w = self.conv.weight.detach()  # [OC, IC, KH, KW]
            w_perm = w.permute(2, 3, 1, 0).contiguous()  # [KH, KW, IC, OC]
            self.register_buffer('w_perm', w_perm.cuda())
            self.register_buffer('bias_buf', self.conv.bias.detach().contiguous().cuda())

    def forward(self, x):
        x = x.cuda()
        if not x.is_contiguous():
            x = x.contiguous()

        B, IC, H, W = x.shape
        KH, KW, _, OC = self.w_perm.shape
        OH = H - KH + 1
        OW = W - KW + 1

        out = torch.empty((B, OC, OH, OW), device=x.device, dtype=x.dtype)
        M = B * OH * OW
        N = OC
        K = KH * KW * IC
        inv_div = 1.0 / self.divisor
        neg_slope = 0.01

        grid = lambda meta: (
            triton.cdiv(N, meta['BLOCK_N']),
            triton.cdiv(M, meta['BLOCK_M']),
        )

        conv2d_im2col_kernel[grid](
            x, self.w_perm, self.bias_buf, out,
            B, H, W, IC,
            OC, KH, KW,
            OH, OW,
            M, N, K,
            inv_div,
            neg_slope,
            IC_C=IC,
        )
        return out