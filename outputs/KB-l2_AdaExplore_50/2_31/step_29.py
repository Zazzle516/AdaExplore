import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_OH': 4, 'BLOCK_OW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OH': 4, 'BLOCK_OW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_OH': 8, 'BLOCK_OW': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OH': 8, 'BLOCK_OW': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OH': 4, 'BLOCK_OW': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OH': 4, 'BLOCK_OW': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_OH': 4, 'BLOCK_OW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_OH': 16, 'BLOCK_OW': 16}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'IC', 'OH', 'OW', 'KH', 'KW'],
)
@triton.jit
def conv2d_nhwc_kernel(
    x_ptr, w_ptr, b_ptr, bias_ptr, out_ptr,
    N, IC, H, W,
    OC, KH, KW,
    OH, OW,
    constant_value, scaling_factor,
    stride_xn, stride_xh, stride_xw, stride_xc,
    stride_wo, stride_wh, stride_ww, stride_wi,
    stride_on, stride_oh, stride_ow, stride_oc,
    BLOCK_OC: tl.constexpr, BLOCK_OH: tl.constexpr, BLOCK_OW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # grid: (n, oc_tile, oh_tile * ow_tile)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_s = tl.program_id(2)

    num_ow_tiles = (OW + BLOCK_OW - 1) // BLOCK_OW
    pid_oh = pid_s // num_ow_tiles
    pid_ow = pid_s % num_ow_tiles

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_oh = pid_oh * BLOCK_OH + tl.arange(0, BLOCK_OH)
    offs_ow = pid_ow * BLOCK_OW + tl.arange(0, BLOCK_OW)

    oc_mask = offs_oc < OC
    oh_mask = offs_oh < OH
    ow_mask = offs_ow < OW

    # Spatial tile flattened: BLOCK_OH * BLOCK_OW positions
    # acc shape: [BLOCK_OC, BLOCK_OH*BLOCK_OW]
    M = BLOCK_OH * BLOCK_OW
    acc = tl.zeros((BLOCK_OC, M), dtype=tl.float32)

    # output spatial index helpers
    sh = tl.arange(0, BLOCK_OH)  # [BLOCK_OH]
    sw = tl.arange(0, BLOCK_OW)  # [BLOCK_OW]
    # flatten: idx = sh * BLOCK_OW + sw
    oh_idx = (pid_oh * BLOCK_OH + sh)[:, None] + tl.zeros((BLOCK_OH, BLOCK_OW), dtype=tl.int32)  # [BH, BW]
    ow_idx = (pid_ow * BLOCK_OW + sw)[None, :] + tl.zeros((BLOCK_OH, BLOCK_OW), dtype=tl.int32)
    oh_flat = tl.reshape(oh_idx, (M,))
    ow_flat = tl.reshape(ow_idx, (M,))
    s_mask = (oh_flat < OH) & (ow_flat < OW)

    offs_ic = tl.arange(0, BLOCK_IC)

    for kh in range(0, KH):
        for kw in range(0, KW):
            ih = oh_flat + kh  # [M]
            iw = ow_flat + kw
            for ic_start in range(0, IC, BLOCK_IC):
                ic = ic_start + offs_ic  # [BLOCK_IC]
                ic_mask = ic < IC

                # Load x[pid_n, ih, iw, ic] : [M, BLOCK_IC]
                x_ptrs = x_ptr + pid_n * stride_xn + ih[:, None] * stride_xh + iw[:, None] * stride_xw + ic[None, :] * stride_xc
                x_m = s_mask[:, None] & ic_mask[None, :]
                x_tile = tl.load(x_ptrs, mask=x_m, other=0.0)  # [M, BLOCK_IC]

                # Load w[oc, kh, kw, ic] : [BLOCK_OC, BLOCK_IC]
                w_ptrs = w_ptr + offs_oc[:, None] * stride_wo + kh * stride_wh + kw * stride_ww + ic[None, :] * stride_wi
                w_m = oc_mask[:, None] & ic_mask[None, :]
                w_tile = tl.load(w_ptrs, mask=w_m, other=0.0)  # [BLOCK_OC, BLOCK_IC]

                # acc[oc, m] += sum_ic w[oc, ic] * x[m, ic]
                # = w @ x^T
                acc += tl.dot(w_tile, tl.trans(x_tile), allow_tf32=False)

    # epilogue
    b_vals = tl.load(b_ptr + offs_oc, mask=oc_mask, other=0.0)
    acc += b_vals[:, None]
    acc = tl.minimum(acc, constant_value)
    extra = tl.load(bias_ptr + offs_oc, mask=oc_mask, other=0.0)
    acc += extra[:, None]
    acc = acc * scaling_factor

    # Store - acc is [BLOCK_OC, M] where M = BLOCK_OH*BLOCK_OW
    # output layout NHWC: out[n, oh, ow, oc]
    out_ptrs = out_ptr + pid_n * stride_on + oh_flat[None, :] * stride_oh + ow_flat[None, :] * stride_ow + offs_oc[:, None] * stride_oc
    out_mask = oc_mask[:, None] & s_mask[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-permute weight to NHWC-friendly: [OC, KH, KW, IC]
        with torch.no_grad():
            w_perm = self.conv.weight.detach().permute(0, 2, 3, 1).contiguous()
        self.register_buffer('w_nhwc', w_perm)

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1

        # Convert input to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        # Weight already NHWC-friendly: [OC, KH, KW, IC]
        # If weights changed (training), re-permute
        if self.training:
            w_nhwc = self.conv.weight.permute(0, 2, 3, 1).contiguous()
        else:
            w_nhwc = self.w_nhwc
            # Safety: if weight was updated, refresh
            if w_nhwc.shape[0] != OC or w_nhwc.shape[3] != IC:
                w_nhwc = self.conv.weight.permute(0, 2, 3, 1).contiguous()

        b = self.conv.bias.contiguous()
        bias_extra = self.bias.contiguous().view(-1)

        # Output NHWC
        out_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

        # Pick BLOCK_IC based on IC
        if IC >= 64:
            BLOCK_IC = 64
        elif IC >= 32:
            BLOCK_IC = 32
        elif IC >= 16:
            BLOCK_IC = 16
        else:
            BLOCK_IC = max(1, triton.next_power_of_2(IC))

        grid = lambda meta: (
            N,
            (OC + meta['BLOCK_OC'] - 1) // meta['BLOCK_OC'],
            ((OH + meta['BLOCK_OH'] - 1) // meta['BLOCK_OH']) * ((OW + meta['BLOCK_OW'] - 1) // meta['BLOCK_OW']),
        )

        conv2d_nhwc_kernel[grid](
            x_nhwc, w_nhwc, b, bias_extra, out_nhwc,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW,
            float(self.constant_value), float(self.scaling_factor),
            x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
            w_nhwc.stride(0), w_nhwc.stride(1), w_nhwc.stride(2), w_nhwc.stride(3),
            out_nhwc.stride(0), out_nhwc.stride(1), out_nhwc.stride(2), out_nhwc.stride(3),
            BLOCK_IC=BLOCK_IC,
        )

        # Convert back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out