import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=2),
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
    IC_CONST: tl.constexpr,
    KH_CONST: tl.constexpr,
    KW_CONST: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < OUT_HW

    oh = offs_m // OW
    ow = offs_m % OW

    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)  # BLOCK_K == IC_CONST

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    x_batch_off = pid_n * H * W * IC_CONST

    # Outer loop over (kh, kw); inner reduction is IC (== BLOCK_K)
    for kh in tl.static_range(0, KH_CONST):
        for kw in tl.static_range(0, KW_CONST):
            ih = oh + kh  # [BLOCK_M]
            iw = ow + kw  # [BLOCK_M]

            x_offset = x_batch_off + ih[:, None] * (W * IC_CONST) + iw[:, None] * IC_CONST + offs_k[None, :]
            x_vals = tl.load(x_ptr + x_offset, mask=m_mask[:, None], other=0.0)

            # weight layout: [KH, KW, IC, OC] flattened
            w_base = (kh * KW_CONST + kw) * IC_CONST * BLOCK_N
            w_offset = w_base + offs_k[:, None] * BLOCK_N + offs_n[None, :]
            w_vals = tl.load(w_ptr + w_offset)

            acc += tl.dot(x_vals, w_vals)

    bias = tl.load(b_ptr + offs_n)
    acc = acc + bias[None, :]
    acc = acc * SCALE

    min_acc = tl.min(acc, axis=1)

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
        self._cached_weight = None
        self._cached_weight_version = None
        self._cached_bias = None

    def _get_weight_packed(self, w):
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

        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        if (self._cached_weight is None or
                self._cached_weight_version != w._version or
                self._cached_weight.device != x.device):
            self._cached_weight = self._get_weight_packed(w.detach()).to(x.device)
            self._cached_bias = b.detach().contiguous().to(x.device)
            self._cached_weight_version = w._version
        wp = self._cached_weight
        bias = self._cached_bias

        out = torch.empty((N, 1, OH, OW), device=x.device, dtype=x.dtype)

        grid = lambda meta: (N, (OUT_HW + meta['BLOCK_M'] - 1) // meta['BLOCK_M'])

        conv_scale_min_kernel_nhwc[grid](
            x_nhwc, wp, bias, out,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW,
            self.scale_factor,
            OUT_HW, IC_KH_KW,
            BLOCK_N=OC,
            BLOCK_K=IC,
            IC_CONST=IC,
            KH_CONST=KH,
            KW_CONST=KW,
        )
        return out