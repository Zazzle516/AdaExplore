import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
    ],
    key=['IC', 'OC', 'OH', 'OW'],
)
@triton.jit
def convt2d_fused_kernel(
    x_ptr, w_ptr, cb_ptr, bias_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    scaling_factor,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)  # full OC axis

    OHOW = OH * OW
    m_mask = offs_m < OHOW
    n_mask = offs_n < OC

    oh = offs_m // OW
    ow = offs_m % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    KHKWOC = KH * KW * OC

    for kh in tl.static_range(0, KH):
        ih_num = oh + PAD - kh
        ih = ih_num // STRIDE
        ih_valid = (ih_num % STRIDE == 0) & (ih >= 0) & (ih < IH)
        for kw in tl.static_range(0, KW):
            iw_num = ow + PAD - kw
            iw = iw_num // STRIDE
            iw_valid = (iw_num % STRIDE == 0) & (iw >= 0) & (iw < IW)
            valid = ih_valid & iw_valid

            ih_safe = tl.where(valid, ih, 0)
            iw_safe = tl.where(valid, iw, 0)
            x_row_base = pid_b * IH * IW * IC + ih_safe * IW * IC + iw_safe * IC

            w_base_ptr = w_ptr + kh * KW * OC + kw * OC + offs_n[None, :]
            vm_mask = valid & m_mask

            for k in range(0, IC, BLOCK_K):
                offs_k = k + tl.arange(0, BLOCK_K)
                k_mask = offs_k < IC

                x_ptrs = x_ptr + x_row_base[:, None] + offs_k[None, :]
                x_full_mask = vm_mask[:, None] & k_mask[None, :]
                x_vals = tl.load(x_ptrs, mask=x_full_mask, other=0.0)

                w_ptrs = w_base_ptr + offs_k[:, None] * KHKWOC
                w_full_mask = k_mask[:, None] & n_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_full_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals)

    # Add conv bias
    cb = tl.load(cb_ptr + offs_n, mask=n_mask, other=0.0)
    acc += cb[None, :]

    # Softmax along channel (axis=1 of acc, BLOCK_N dim)
    acc_masked = tl.where(n_mask[None, :], acc, -float('inf'))
    max_val = tl.max(acc_masked, axis=1)  # [BLOCK_M]
    e = tl.exp(acc_masked - max_val[:, None])
    e = tl.where(n_mask[None, :], e, 0.0)
    sum_e = tl.sum(e, axis=1)  # [BLOCK_M]
    sm = e / sum_e[:, None]

    # add bias, scale, sigmoid
    bias_vals = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
    y = (sm + bias_vals[None, :]) * scaling_factor
    out_vals = 1.0 / (1.0 + tl.exp(-y))

    # Store as NHWC: out[n, oh, ow, c] => pid_b*OHOW*OC + offs_m*OC + offs_n
    out_ptrs = out_ptr + pid_b * OHOW * OC + offs_m[:, None] * OC + offs_n[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptrs, out_vals, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = float(scaling_factor)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        self._cached_w = None
        self._cached_w_version = None

    def _get_weight_reshaped(self):
        w = self.conv_transpose.weight
        if (self._cached_w is None) or (self._cached_w_version != w._version) or self.training:
            self._cached_w = w.detach().permute(0, 2, 3, 1).contiguous()
            self._cached_w_version = w._version
        return self._cached_w

    def forward(self, x):
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        stride = self.stride
        pad = self.padding
        opad = self.output_padding

        OH = (IH - 1) * stride - 2 * pad + KH + opad
        OW = (IW - 1) * stride - 2 * pad + KW + opad

        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        w_reshaped = self._get_weight_reshaped()

        conv_bias = self.conv_transpose.bias
        if conv_bias is None:
            conv_bias = torch.zeros(OC, device=x.device, dtype=torch.float32)
        else:
            conv_bias = conv_bias.contiguous()

        bias_flat = self.bias.view(-1).contiguous()

        out_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=torch.float32)

        BLOCK_N = triton.next_power_of_2(OC)

        OHOW = OH * OW
        grid = lambda META: (triton.cdiv(OHOW, META['BLOCK_M']), N)

        convt2d_fused_kernel[grid](
            x_nhwc, w_reshaped, conv_bias, bias_flat, out_nhwc,
            N, IC, IH, IW,
            OC, OH, OW,
            self.scaling_factor,
            KH, KW,
            stride, pad,
            BLOCK_N=BLOCK_N,
        )
        return out_nhwc.permute(0, 3, 1, 2).contiguous()