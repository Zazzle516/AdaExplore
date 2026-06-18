import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
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
):
    pid_n = tl.program_id(0)  # batch index
    pid_m = tl.program_id(1)  # tile along output spatial (OH*OW)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # spatial output positions
    m_mask = offs_m < OUT_HW

    oh = offs_m // OW
    ow = offs_m % OW

    INF = float('inf')
    min_acc = tl.full([BLOCK_M], INF, dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    x_batch_off = pid_n * H * W * IC

    num_oc_tiles = (OC + BLOCK_N - 1) // BLOCK_N
    for oc_tile in range(0, num_oc_tiles):
        offs_n = oc_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < OC

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        # Loop over (kh, kw, ic_chunk) with ic contiguous (coalesced NHWC)
        for kh in range(0, KH):
            for kw in range(0, KW):
                ih = oh + kh  # [BLOCK_M]
                iw = ow + kw  # [BLOCK_M]
                x_row_base = x_batch_off + ih * (W * IC) + iw * IC  # [BLOCK_M]
                w_kh_kw_base = (kh * KW + kw) * IC  # scalar

                for ic_start in range(0, IC, BLOCK_K):
                    ic_idx = ic_start + offs_k  # [BLOCK_K]
                    ic_mask = ic_idx < IC

                    x_offset = x_row_base[:, None] + ic_idx[None, :]
                    x_load_mask = m_mask[:, None] & ic_mask[None, :]
                    x_vals = tl.load(x_ptr + x_offset, mask=x_load_mask, other=0.0)

                    w_offset = (w_kh_kw_base + ic_idx)[:, None] * OC + offs_n[None, :]
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
        # Pre-permute weight to layout [KH, KW, IC, OC] flattened as [KH*KW*IC, OC]
        self._cached_weight = None
        self._cached_weight_version = None

    def _get_weight_packed(self, w):
        # w: [OC, IC, KH, KW] -> [KH, KW, IC, OC] -> [KH*KW*IC, OC]
        OC, IC, KH, KW = w.shape
        wp = w.permute(2, 3, 1, 0).contiguous().view(KH * KW * IC, OC)
        return wp

    def forward(self, x):
        x = x.cuda(non_blocking=True)
        w = self.conv.weight
        b = self.conv.bias

        N, IC, H, W = x.shape
        OC, _, KH, KW = w.shape
        OH = H - KH + 1
        OW = W - KW + 1
        OUT_HW = OH * OW
        IC_KH_KW = IC * KH * KW

        # NHWC permute
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        # weight packed (cache across calls)
        if (self._cached_weight is None or
                self._cached_weight_version != w._version or
                self._cached_weight.device != w.device):
            self._cached_weight = self._get_weight_packed(w.detach()).to(x.device)
            self._cached_weight_version = w._version
        wp = self._cached_weight

        bias = b.contiguous().to(x.device)

        out = torch.empty((N, 1, OH, OW), device=x.device, dtype=x.dtype)

        grid = lambda meta: (N, (OUT_HW + meta['BLOCK_M'] - 1) // meta['BLOCK_M'])

        conv_scale_min_kernel_nhwc[grid](
            x_nhwc, wp, bias, out,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW,
            self.scale_factor,
            OUT_HW, IC_KH_KW,
        )
        return out