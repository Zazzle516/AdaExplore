import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OW': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_OW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 256}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'OC', 'OH', 'OW'],
)
@triton.jit
def _conv_transpose2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_OW: tl.constexpr,
):
    pid_ow_full = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_nh = tl.program_id(2)

    # Decode (n, oh) from pid_nh
    n_id = pid_nh // OH
    oh = pid_nh % OH

    # Decode ow-parity and ow-tile from pid_ow_full
    ow_par = pid_ow_full % STRIDE
    ow_tile = pid_ow_full // STRIDE

    # ow positions within this parity: strided by STRIDE in actual ow space
    ow_idx_in_par = ow_tile * BLOCK_OW + tl.arange(0, BLOCK_OW)  # [BLOCK_OW]
    ow_offs = ow_par + ow_idx_in_par * STRIDE
    ow_mask = ow_offs < OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Parity-aware kernel taps: only kh with (oh+PAD-kh) % STRIDE == 0 contribute,
    # i.e. kh ≡ (oh+PAD) mod STRIDE. Same for kw.
    kh_par = (oh + PAD) % STRIDE        # scalar
    kw_par = (ow_par + PAD) % STRIDE    # scalar

    ic_offs = tl.arange(0, IC)

    acc = tl.zeros((BLOCK_OW, BLOCK_OC), dtype=tl.float32)

    x_base = n_id * IC * IH * IW

    for kh_idx in tl.static_range(0, KH // STRIDE):
        kh = kh_par + kh_idx * STRIDE
        ih = (oh + PAD - kh) // STRIDE  # scalar
        valid_h = (ih >= 0) & (ih < IH)
        for kw_idx in tl.static_range(0, KW // STRIDE):
            kw = kw_par + kw_idx * STRIDE
            iw = (ow_offs + PAD - kw) // STRIDE  # [BLOCK_OW]
            valid = valid_h & (iw >= 0) & (iw < IW) & ow_mask

            # Gather x[n_id, :, ih, iw]: shape [BLOCK_OW, IC]
            x_off = x_base + ic_offs[None, :] * (IH * IW) + ih * IW + iw[:, None]
            x_tile = tl.load(x_ptr + x_off, mask=valid[:, None], other=0.0)

            # Pre-transposed weight w_t[oc, ic, kh, kw]: shape [IC, BLOCK_OC]
            w_off = oc_offs[None, :] * (IC * KH * KW) + ic_offs[:, None] * (KH * KW) + kh * KW + kw
            w_tile = tl.load(w_ptr + w_off, mask=oc_mask[None, :], other=0.0)

            acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + b_val[None, :]
    e2x = tl.exp(2.0 * acc)
    out = (e2x - 1.0) / (e2x + 1.0)

    out_off = n_id * (OC * OH * OW) + oc_offs[None, :] * (OH * OW) + oh * OW + ow_offs[:, None]
    mask = ow_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        # Use the same init as nn.ConvTranspose2d
        conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                  stride=stride, padding=padding, output_padding=output_padding)
        # weight shape: [IC, OC, KH, KW]
        self.conv_weight = nn.Parameter(conv.weight.detach().clone())
        self.conv_bias = nn.Parameter(conv.bias.detach().clone())
        self.bias = nn.Parameter(torch.randn(bias_shape))

        # Cached pre-transposed weight: [OC, IC, KH, KW]
        self._weight_t_cache = None
        self._weight_version = -1

    def _get_weight_t(self):
        # Permute [IC, OC, KH, KW] -> [OC, IC, KH, KW], cache across calls
        v = self.conv_weight._version
        if self._weight_t_cache is None or self._weight_version != v or \
           self._weight_t_cache.device != self.conv_weight.device:
            self._weight_t_cache = self.conv_weight.detach().permute(1, 0, 2, 3).contiguous()
            self._weight_version = v
        return self._weight_t_cache

    def forward(self, x):
        x = x.contiguous()
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        stride = self.stride
        pad = self.padding
        outpad = self.output_padding

        OH = (IH - 1) * stride - 2 * pad + KH + outpad
        OW = (IW - 1) * stride - 2 * pad + KW + outpad

        # Effective conv bias: conv_bias - self.bias (broadcast over OC)
        eff_bias = (self.conv_bias - self.bias.view(-1)).contiguous()

        weight_t = self._get_weight_t()

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        # Per-parity ow tile count (max over parities)
        num_ow_per_par = (OW + stride - 1) // stride

        grid = lambda META: (
            stride * ((num_ow_per_par + META['BLOCK_OW'] - 1) // META['BLOCK_OW']),
            (OC + META['BLOCK_OC'] - 1) // META['BLOCK_OC'],
            N * OH,
        )

        _conv_transpose2d_kernel[grid](
            x, weight_t, eff_bias, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH=KH, KW=KW,
            STRIDE=stride, PAD=pad,
        )

        return out