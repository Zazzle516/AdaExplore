import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'OC', 'H_OUT', 'W_OUT', 'KH', 'KW'],
)
@triton.jit
def conv_hardswish_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, H, W,
    OC, H_OUT, W_OUT,
    KH: tl.constexpr, KW: tl.constexpr,
    K: tl.constexpr,  # IC*KH*KW
    K_PAD: tl.constexpr,  # next pow2 of K
    stride_xn, stride_xh, stride_xw,  # NHWC strides for x (channel stride = 1)
    stride_on, stride_oh, stride_ow,  # NCHW strides for output
    BLOCK_OC: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_k = tl.arange(0, K_PAD)

    mask_oc = offs_oc < OC
    mask_hw = offs_hw < (H_OUT * W_OUT)
    mask_k = offs_k < K

    oh = offs_hw // W_OUT
    ow = offs_hw % W_OUT

    # Decompose k -> (kh, kw, ic) where x is NHWC and weight is packed as [OC, KH, KW, IC]
    offs_k_safe = tl.where(mask_k, offs_k, 0)
    kh = (offs_k_safe // KW) // IC
    kw = (offs_k_safe // IC) % KW
    ic = offs_k_safe % IC

    # Precompute K-vector offset within an input window for given (kh, kw, ic)
    k_off = kh * stride_xh + kw * stride_xw + ic  # [K_PAD]

    # Weight packed as [OC, K] row-major
    w_ptrs = w_ptr + offs_oc[:, None] * K + offs_k[None, :]
    w_vals = tl.load(w_ptrs, mask=mask_oc[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_OC, K_PAD]

    # Input gather: x is NHWC
    base = pid_n * stride_xn + oh * stride_xh + ow * stride_xw  # [BLOCK_HW]
    x_off = base[:, None] + k_off[None, :]
    x_vals = tl.load(x_ptr + x_off, mask=mask_hw[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_HW, K_PAD]

    # GEMM: [BLOCK_OC, K] x [K, BLOCK_HW] -> [BLOCK_OC, BLOCK_HW]
    acc = tl.dot(w_vals, tl.trans(x_vals))

    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc += bias[:, None]

    x_plus_3 = acc + 3.0
    relu6 = tl.minimum(tl.maximum(x_plus_3, 0.0), 6.0)
    hs = acc * relu6 * (1.0 / 6.0)
    out = tl.maximum(hs, 0.0)

    out_off = (pid_n * stride_on
               + offs_oc[:, None] * (H_OUT * W_OUT)
               + oh[None, :] * stride_oh
               + ow[None, :] * stride_ow)
    mask = mask_oc[:, None] & mask_hw[None, :]
    tl.store(out_ptr + out_off, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-pack weight as [OC, KH, KW, IC] (matches NHWC input gather order)
        with torch.no_grad():
            w = self.conv.weight.detach()  # [OC, IC, KH, KW]
            w_packed = w.permute(0, 2, 3, 1).contiguous()  # [OC, KH, KW, IC]
            OC = w.shape[0]
            self.register_buffer('weight_packed', w_packed.view(OC, -1).contiguous())
            self.register_buffer('bias_packed', self.conv.bias.detach().contiguous())

        self.KH = kernel_size
        self.KW = kernel_size
        self.K = in_channels * kernel_size * kernel_size

    def forward(self, x):
        x = x.cuda()
        # Convert to NHWC contiguous
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [N, H, W, IC]
        N, H, W, IC = x_nhwc.shape
        OC = self.out_channels
        KH = self.KH
        KW = self.KW
        H_OUT = H - KH + 1
        W_OUT = W - KW + 1

        out = torch.empty((N, OC, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

        # NHWC strides
        stride_xn = H * W * IC
        stride_xh = W * IC
        stride_xw = IC

        stride_on = OC * H_OUT * W_OUT
        stride_oh = W_OUT
        stride_ow = 1

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_OC']),
            triton.cdiv(H_OUT * W_OUT, meta['BLOCK_HW']),
        )

        K = self.K
        K_PAD = 1
        while K_PAD < K:
            K_PAD *= 2

        conv_hardswish_relu_kernel[grid](
            x_nhwc, self.weight_packed, self.bias_packed, out,
            N, IC, H, W,
            OC, H_OUT, W_OUT,
            KH, KW, K, K_PAD,
            stride_xn, stride_xh, stride_xw,
            stride_on, stride_oh, stride_ow,
        )
        return out