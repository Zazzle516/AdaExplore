import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 32, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 32, 'BLOCK_OC': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 32, 'BLOCK_OC': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 16, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 16, 'BLOCK_OC': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 16, 'BLOCK_OC': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 64, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 64, 'BLOCK_OC': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 64, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 64, 'BLOCK_OC': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 32, 'BLOCK_OW': 8, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 32, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
    ],
    key=['OH', 'OW', 'OC', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv_relu_hardswish_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, IH, IW,
    OC, OH, OW,
    IC: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wh, stride_ww,
    stride_yb, stride_yc, stride_yh, stride_yw,
    BLOCK_OH: tl.constexpr, BLOCK_OW: tl.constexpr, BLOCK_OC: tl.constexpr,
):
    pid_spatial = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_b = tl.program_id(2)

    num_tiles_w = tl.cdiv(OW, BLOCK_OW)
    tile_h = pid_spatial // num_tiles_w
    tile_w = pid_spatial % num_tiles_w

    offs_oh = tile_h * BLOCK_OH + tl.arange(0, BLOCK_OH)
    offs_ow = tile_w * BLOCK_OW + tl.arange(0, BLOCK_OW)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    mask_oh = offs_oh < OH
    mask_ow = offs_ow < OW
    mask_oc = offs_oc < OC

    # Flatten N = OH * OW within tile
    N_TILE: tl.constexpr = BLOCK_OH * BLOCK_OW
    n_idx = tl.arange(0, N_TILE)
    n_oh = n_idx // BLOCK_OW
    n_ow = n_idx % BLOCK_OW
    cur_oh = tile_h * BLOCK_OH + n_oh  # [N_TILE]
    cur_ow = tile_w * BLOCK_OW + n_ow  # [N_TILE]
    mask_n = (cur_oh < OH) & (cur_ow < OW)

    base_x = pid_b * stride_xb + cur_oh * stride_xh + cur_ow * stride_xw  # [N_TILE]

    acc = tl.zeros((N_TILE, BLOCK_OC), dtype=tl.float32)

    # Loop over IC, KH, KW, blocking K=IC dimension as inner loop
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            # for each (kh, kw), iterate ic
            x_off_kh_kw = kh * stride_xh + kw * stride_xw  # scalar
            w_off_kh_kw = kh * stride_wh + kw * stride_ww  # scalar
            ic_range = tl.arange(0, IC)
            # load x: shape [N_TILE, IC]
            x_offs = (base_x[:, None] + x_off_kh_kw + ic_range[None, :] * stride_xc)
            x_tile = tl.load(x_ptr + x_offs, mask=mask_n[:, None], other=0.0)
            # load w: shape [IC, BLOCK_OC]
            w_offs = (offs_oc[None, :] * stride_wo + ic_range[:, None] * stride_wi + w_off_kh_kw)
            w_tile = tl.load(w_ptr + w_offs, mask=mask_oc[None, :], other=0.0)
            acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + bias[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)
    # HardSwish: x * clamp((x+3)/6, 0, 1)
    hs = (acc + 3.0) * (1.0 / 6.0)
    hs = tl.minimum(tl.maximum(hs, 0.0), 1.0)
    out = acc * hs

    # Store
    y_offs = (pid_b * stride_yb + offs_oc[None, :] * stride_yc +
              cur_oh[:, None] * stride_yh + cur_ow[:, None] * stride_yw)
    mask_store = mask_n[:, None] & mask_oc[None, :]
    tl.store(y_ptr + y_offs, out, mask=mask_store)


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

        B, IC, IH, IW = x.shape
        OC, _, KH, KW = w.shape
        OH = IH - KH + 1
        OW = IW - KW + 1

        y = torch.empty((B, OC, OH, OW), device=x.device, dtype=x.dtype)

        grid = lambda meta: (
            triton.cdiv(OH, meta['BLOCK_OH']) * triton.cdiv(OW, meta['BLOCK_OW']),
            triton.cdiv(OC, meta['BLOCK_OC']),
            B,
        )

        conv_relu_hardswish_kernel[grid](
            x, w, b, y,
            B, IH, IW,
            OC, OH, OW,
            IC, KH, KW,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w.stride(0), w.stride(1), w.stride(2), w.stride(3),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        )
        return y