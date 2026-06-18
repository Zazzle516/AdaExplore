import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=3),
    ],
    key=['IC', 'OC', 'H_OUT', 'W_OUT', 'KH', 'KW'],
)
@triton.jit
def conv2d_hswish_relu_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, H, W,
    H_OUT, W_OUT,
    IC: tl.constexpr, OC: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    stride_xn, stride_xh, stride_xw,  # NHWC strides; xc stride = 1
    stride_yn, stride_yh, stride_yw,  # NHWC strides; yc stride = 1
    BLOCK_HW: tl.constexpr,
    K: tl.constexpr,  # IC*KH*KW
):
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)

    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offs < (H_OUT * W_OUT)

    oh = hw_offs // W_OUT
    ow = hw_offs % W_OUT

    oc_offs = tl.arange(0, OC)
    k_offs = tl.arange(0, K)  # K = IC*KH*KW

    # Decompose k -> (ic, kh, kw): k = ((ic)*KH + kh)*KW + kw
    kw_idx = k_offs % KW
    tmp = k_offs // KW
    kh_idx = tmp % KH
    ic_idx = tmp // KH

    # Weight is laid out as [OC, IC, KH, KW] contiguous, so flat index over IC*KH*KW
    # works directly: w[oc, k] = w_ptr[oc * K + k]
    w_off = oc_offs[:, None] * K + k_offs[None, :]
    w_tile = tl.load(w_ptr + w_off)  # [OC, K]

    # Input is NHWC: x[n, h, w, c] -> x_ptr + n*stride_xn + h*stride_xh + w*stride_xw + c
    # For each output pixel hw and each k=(ic,kh,kw):
    #   in_h = oh + kh, in_w = ow + kw
    # x_off = pid_n*stride_xn + in_h*stride_xh + in_w*stride_xw + ic
    in_h = oh[:, None] + kh_idx[None, :]  # [BLOCK_HW, K]
    in_w = ow[:, None] + kw_idx[None, :]
    x_off = pid_n * stride_xn + in_h * stride_xh + in_w * stride_xw + ic_idx[None, :]

    x_tile = tl.load(x_ptr + x_off, mask=hw_mask[:, None], other=0.0)  # [BLOCK_HW, K]

    # GEMM: [OC, K] x [K, BLOCK_HW] -> [OC, BLOCK_HW]
    # We have w_tile [OC, K] and x_tile [BLOCK_HW, K], so compute via dot(w, x^T)
    acc = tl.dot(w_tile, tl.trans(x_tile), allow_tf32=True)  # [OC, BLOCK_HW]

    bias = tl.load(b_ptr + oc_offs)
    acc = acc + bias[:, None]

    t = acc + 3.0
    t = tl.maximum(t, 0.0)
    t = tl.minimum(t, 6.0)
    out = acc * t * (1.0 / 6.0)
    out = tl.maximum(out, 0.0)

    # Output NHWC: y[n, oh, ow, oc]
    y_off = (pid_n * stride_yn
             + oh[None, :] * stride_yh
             + ow[None, :] * stride_yw
             + oc_offs[:, None])
    tl.store(y_ptr + y_off, out, mask=hw_mask[None, :])


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # Pre-store weight in [OC, IC, KH, KW] contiguous form (already default)
        # We'll keep weight as a buffer for fast access
        self.register_buffer('_w', self.conv.weight.detach().contiguous().cuda(), persistent=False)
        self.register_buffer('_b', self.conv.bias.detach().contiguous().cuda(), persistent=False)

    def forward(self, x):
        x = x.cuda()
        # Convert to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        N, H, W, IC = x_nhwc.shape
        OC = self._w.shape[0]
        KH = self._w.shape[2]
        KW = self._w.shape[3]
        H_OUT = H - KH + 1
        W_OUT = W - KW + 1

        # Output in NHWC
        y_nhwc = torch.empty((N, H_OUT, W_OUT, OC), device=x.device, dtype=x.dtype)

        K = IC * KH * KW

        grid = lambda meta: (
            N,
            triton.cdiv(H_OUT * W_OUT, meta['BLOCK_HW']),
        )

        conv2d_hswish_relu_kernel[grid](
            x_nhwc, self._w, self._b, y_nhwc,
            N, H, W,
            H_OUT, W_OUT,
            IC, OC,
            KH, KW,
            x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2),
            y_nhwc.stride(0), y_nhwc.stride(1), y_nhwc.stride(2),
            K=K,
        )

        # Convert back to NCHW
        y = y_nhwc.permute(0, 3, 1, 2).contiguous()
        return y