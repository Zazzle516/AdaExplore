import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=8, num_stages=3),
    ],
    key=['IC', 'OC', 'H_OUT', 'W_OUT', 'KH', 'KW'],
)
@triton.jit
def conv2d_hswish_relu_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, IC, H, W,
    OC, H_OUT, W_OUT,
    KH: tl.constexpr, KW: tl.constexpr,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_OC: tl.constexpr, BLOCK_HW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    ic_offs = tl.arange(0, BLOCK_IC)

    oc_mask = oc_offs < OC
    hw_mask = hw_offs < (H_OUT * W_OUT)
    ic_mask = ic_offs < IC

    oh = hw_offs // W_OUT
    ow = hw_offs % W_OUT

    # Precompute x base offsets for spatial positions
    x_base_n = pid_n * stride_xn
    x_spatial = oh * stride_xh + ow * stride_xw  # [BLOCK_HW]

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            # weight tile [BLOCK_OC, BLOCK_IC]
            w_off = (oc_offs[:, None] * stride_wo
                     + ic_offs[None, :] * stride_wi
                     + kh * stride_wh + kw * stride_ww)
            w_tile = tl.load(w_ptr + w_off,
                             mask=oc_mask[:, None] & ic_mask[None, :],
                             other=0.0)
            # input tile [BLOCK_IC, BLOCK_HW]
            x_off = (x_base_n
                     + ic_offs[:, None] * stride_xc
                     + (x_spatial + kh * stride_xh + kw * stride_xw)[None, :])
            x_tile = tl.load(x_ptr + x_off,
                             mask=ic_mask[:, None] & hw_mask[None, :],
                             other=0.0)
            acc += tl.dot(w_tile, x_tile, allow_tf32=True)

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[:, None]

    # HardSwish: x * relu6(x+3) / 6, then ReLU
    # Combined: max(0, x * min(max(x+3, 0), 6) / 6)
    t = acc + 3.0
    t = tl.maximum(t, 0.0)
    t = tl.minimum(t, 6.0)
    out = acc * t * (1.0 / 6.0)
    out = tl.maximum(out, 0.0)

    y_off = pid_n * stride_yn + oc_offs[:, None] * stride_yc + oh[None, :] * stride_yh + ow[None, :] * stride_yw
    mask = oc_mask[:, None] & hw_mask[None, :]
    tl.store(y_ptr + y_off, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.cuda().contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()

        N, IC, H, W = x.shape
        OC, _, KH, KW = w.shape
        H_OUT = H - KH + 1
        W_OUT = W - KW + 1

        y = torch.empty((N, OC, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_OC']),
            triton.cdiv(H_OUT * W_OUT, meta['BLOCK_HW']),
        )

        conv2d_hswish_relu_kernel[grid](
            x, w, b, y,
            N, IC, H, W,
            OC, H_OUT, W_OUT,
            KH, KW,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w.stride(0), w.stride(1), w.stride(2), w.stride(3),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            BLOCK_IC=16,
        )
        return y