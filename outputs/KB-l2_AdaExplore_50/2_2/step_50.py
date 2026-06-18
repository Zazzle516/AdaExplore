import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'BLOCK_IC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_IC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64, 'BLOCK_IC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 128, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 128, 'BLOCK_IC': 64}, num_warps=8, num_stages=3),
    ],
    key=['N', 'IC', 'OC', 'H', 'W', 'OH', 'OW'],
)
@triton.jit
def conv_transpose2d_gemm_kernel(
    x_ptr,           # input (N, IC, H, W)
    w_ptr,           # weight pre-transposed: (KH, KW, IC, OC) contiguous
    b_ptr,           # bias (OC,)
    out_ptr,         # output (N, OC, OH, OW)
    N, IC, OC, H, W, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    SCALE: tl.constexpr, INV_SCALE: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_SP: tl.constexpr, BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]

    oh = sp_offs // OW
    ow = sp_offs % OW

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < OH * OW

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    ic_range = tl.arange(0, BLOCK_IC)  # [BLOCK_IC]

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih_num = oh + PAD - kh
            iw_num = ow + PAD - kw
            ih = ih_num // STRIDE
            iw = iw_num // STRIDE
            valid_h = (ih_num % STRIDE == 0) & (ih >= 0) & (ih < H)
            valid_w = (iw_num % STRIDE == 0) & (iw >= 0) & (iw < W)
            valid = valid_h & valid_w & sp_mask  # [BLOCK_SP]

            # weight tile [BLOCK_IC, BLOCK_OC] for (kh, kw)
            # w layout: (KH, KW, IC, OC), offset = ((kh*KW + kw)*IC + ic)*OC + oc
            for ic_start in range(0, IC, BLOCK_IC):
                ic_offs = ic_start + ic_range  # [BLOCK_IC]
                ic_mask = ic_offs < IC

                # w_offs: [BLOCK_IC, BLOCK_OC]
                w_offs = ((kh * KW + kw) * IC + ic_offs[:, None]) * OC + oc_offs[None, :]
                w_mask = ic_mask[:, None] & oc_mask[None, :]
                w_tile = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)  # [BLOCK_IC, BLOCK_OC]

                # x tile [BLOCK_SP, BLOCK_IC]
                # offset = ((pid_n*IC + ic)*H + ih)*W + iw
                x_offs = ((pid_n * IC + ic_offs[None, :]) * H + ih[:, None]) * W + iw[:, None]
                x_mask = valid[:, None] & ic_mask[None, :]
                x_tile = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)  # [BLOCK_SP, BLOCK_IC]

                acc += tl.dot(x_tile, w_tile)

    # bias
    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc = acc + b_val[None, :]

    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    acc = acc * SCALE
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    acc = acc * INV_SCALE

    # write out: (N, OC, OH, OW)
    out_offs = ((pid_n * OC + oc_offs[None, :]) * OH * OW) + sp_offs[:, None]
    mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_offs, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self._cached_weight = None
        self._cached_bias = None
        self._cache_device = None

    def _ensure_cache(self, device):
        if self._cached_weight is None or self._cache_device != device:
            w = self.conv_transpose.weight.detach().permute(2, 3, 0, 1).contiguous().to(device)
            b = (self.conv_transpose.bias.detach() + self.bias.detach().view(-1)).contiguous().to(device)
            self._cached_weight = w
            self._cached_bias = b
            self._cache_device = device
        return self._cached_weight, self._cached_bias

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = (H - 1) * self.stride - 2 * self.padding + KH + self.output_padding
        OW = (W - 1) * self.stride - 2 * self.padding + KW + self.output_padding

        weight, combined_bias = self._ensure_cache(x.device)

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        grid = lambda META: (N, triton.cdiv(OC, META['BLOCK_OC']), triton.cdiv(OH * OW, META['BLOCK_SP']))

        conv_transpose2d_gemm_kernel[grid](
            x, weight, combined_bias, out,
            N, IC, OC, H, W, OH, OW,
            KH, KW,
            self.stride, self.padding,
            float(self.scaling_factor), float(1.0 / self.scaling_factor),
        )
        return out