import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OUT_HW', 'IC_KH_KW'],
)
@triton.jit
def conv_scale_min_kernel_nhwc(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, KH, KW,
    OH, OW,
    SCALE,
    OUT_HW, IC_KH_KW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SINGLE_OC_TILE: tl.constexpr,
    OC_DIVISIBLE: tl.constexpr,
    KH_CONST: tl.constexpr,
    KW_CONST: tl.constexpr,
):
    pid_n = tl.program_id(0)  # batch index
    pid_m = tl.program_id(1)  # tile along output spatial (OH*OW)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # spatial output positions
    m_mask = offs_m < OUT_HW

    oh = offs_m // OW
    ow = offs_m % OW

    INF = float('inf')

    offs_k = tl.arange(0, BLOCK_K)
    offs_n_base = tl.arange(0, BLOCK_N)

    # Per-M base offset into x: n*H*W*IC + oh*W*IC + ow*IC
    W_IC = W * IC
    x_base_m = pid_n * H * W_IC + oh * W_IC + ow * IC  # [BLOCK_M]
    IC_OC = IC * OC

    if SINGLE_OC_TILE:
        offs_n = offs_n_base
        n_mask = offs_n < OC

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        # Explicit unrolled kh, kw loops; reduce only over ic (chunked by BLOCK_K)
        for kh in tl.static_range(0, KH_CONST):
            for kw in tl.static_range(0, KW_CONST):
                x_base_khkw = x_base_m + kh * W_IC + kw * IC  # [BLOCK_M]
                w_base_khkw = (kh * KW_CONST + kw) * IC_OC    # scalar

                for ic_start in range(0, IC, BLOCK_K):
                    ic = ic_start + offs_k  # [BLOCK_K]
                    ic_mask = ic < IC

                    x_offset = x_base_khkw[:, None] + ic[None, :]
                    x_load_mask = m_mask[:, None] & ic_mask[None, :]
                    x_vals = tl.load(x_ptr + x_offset, mask=x_load_mask, other=0.0)

                    w_offset = w_base_khkw + ic[:, None] * OC + offs_n[None, :]
                    if OC_DIVISIBLE:
                        w_load_mask = ic_mask[:, None]
                    else:
                        w_load_mask = ic_mask[:, None] & n_mask[None, :]
                    w_vals = tl.load(w_ptr + w_offset, mask=w_load_mask, other=0.0)

                    acc += tl.dot(x_vals, w_vals)

        if OC_DIVISIBLE:
            bias = tl.load(b_ptr + offs_n)
        else:
            bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
        acc = acc + bias[None, :]
        acc = acc * SCALE
        if OC_DIVISIBLE:
            min_acc = tl.min(acc, axis=1)
        else:
            acc = tl.where(n_mask[None, :], acc, INF)
            min_acc = tl.min(acc, axis=1)
    else:
        min_acc = tl.full([BLOCK_M], INF, dtype=tl.float32)
        num_oc_tiles = (OC + BLOCK_N - 1) // BLOCK_N

        for oc_tile in range(0, num_oc_tiles):
            offs_n = oc_tile * BLOCK_N + offs_n_base
            n_mask = offs_n < OC

            acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

            for kh in tl.static_range(0, KH_CONST):
                for kw in tl.static_range(0, KW_CONST):
                    x_base_khkw = x_base_m + kh * W_IC + kw * IC
                    w_base_khkw = (kh * KW_CONST + kw) * IC_OC

                    for ic_start in range(0, IC, BLOCK_K):
                        ic = ic_start + offs_k
                        ic_mask = ic < IC

                        x_offset = x_base_khkw[:, None] + ic[None, :]
                        x_load_mask = m_mask[:, None] & ic_mask[None, :]
                        x_vals = tl.load(x_ptr + x_offset, mask=x_load_mask, other=0.0)

                        w_offset = w_base_khkw + ic[:, None] * OC + offs_n[None, :]
                        w_load_mask = ic_mask[:, None] & n_mask[None, :]
                        w_vals = tl.load(w_ptr + w_offset, mask=w_load_mask, other=0.0)

                        acc += tl.dot(x_vals, w_vals)

            bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
            acc = acc + bias[None, :]
            acc = acc * SCALE
            acc = tl.where(n_mask[None, :], acc, INF)
            tile_min = tl.min(acc, axis=1)
            min_acc = tl.minimum(min_acc, tile_min)

    out_offset = pid_n * OUT_HW + offs_m
    tl.store(out_ptr + out_offset, min_acc, mask=m_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = float(scale_factor)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-permute weight to (KH, KW, IC, OC) layout for NHWC-friendly K loop
        with torch.no_grad():
            w = self.conv.weight.detach()  # (OC, IC, KH, KW)
            w_nhwc = w.permute(2, 3, 1, 0).contiguous()  # (KH, KW, IC, OC)
        self.register_buffer('w_nhwc', w_nhwc, persistent=False)

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, H, W = x.shape
        # Permute input to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        w_nhwc = self.w_nhwc
        if w_nhwc.device != x.device:
            w_nhwc = w_nhwc.to(x.device)
            self.w_nhwc = w_nhwc

        b = self.conv.bias.contiguous().to(x.device)

        OC = self.out_channels
        KH = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0]
        KW = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[1]
        OH = H - KH + 1
        OW = W - KW + 1
        OUT_HW = OH * OW
        IC_KH_KW = IC * KH * KW

        out = torch.empty((N, 1, OH, OW), device=x.device, dtype=x.dtype)

        grid = lambda meta: (N, (OUT_HW + meta['BLOCK_M'] - 1) // meta['BLOCK_M'])

        single_oc_tile = (OC <= 128)

        conv_scale_min_kernel_nhwc[grid](
            x_nhwc, w_nhwc, b, out,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW,
            self.scale_factor,
            OUT_HW, IC_KH_KW,
            SINGLE_OC_TILE=single_oc_tile,
            OC_DIVISIBLE=(OC % 128 == 0),
            KH_CONST=KH,
            KW_CONST=KW,
        )
        return out