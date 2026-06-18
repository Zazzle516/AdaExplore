import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['N', 'C_in', 'H_in', 'W_in', 'C_out', 'KH', 'KW'],
)
@triton.jit
def conv_mish_bn_nhwc_kernel(
    x_ptr,        # [N, H_in, W_in, C_in] NHWC
    w_ptr,        # [C_out, KH, KW, C_in]  (OC, KH, KW, IC) - contiguous
    b_ptr,        # [C_out]
    scale_ptr,    # [C_out]
    shift_ptr,    # [C_out]
    out_ptr,      # [N, H_out, W_out, C_out] NHWC
    N, C_in, H_in, W_in,
    C_out, KH, KW,
    H_out, W_out,
    BLOCK_M: tl.constexpr,   # output spatial tile
    BLOCK_N: tl.constexpr,   # OC tile (contiguous inner dim for store)
    BLOCK_K: tl.constexpr,   # IC tile (contiguous in NHWC)
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    HW_out = H_out * W_out

    offs_oc = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_sp = pid_sp * BLOCK_M + tl.arange(0, BLOCK_M)

    mask_oc = offs_oc < C_out
    mask_sp = offs_sp < HW_out

    oh = offs_sp // W_out
    ow = offs_sp % W_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    HW_in = H_in * W_in
    x_batch_off = pid_n * HW_in * C_in
    KHKW_Cin = KH * KW * C_in
    w_base = offs_oc[:, None] * KHKW_Cin  # [BLOCK_N, 1]

    # Loop over (kh, kw) and IC tiles
    for kh in range(0, KH):
        ih_W = (oh + kh) * W_in  # [BLOCK_M]
        for kw in range(0, KW):
            iw = ow + kw  # [BLOCK_M]
            spatial_off = (ih_W + iw) * C_in  # [BLOCK_M]
            kw_base = (kh * KW + kw) * C_in

            for ic_start in range(0, C_in, BLOCK_K):
                offs_k = ic_start + tl.arange(0, BLOCK_K)
                mask_k = offs_k < C_in

                # x: [BLOCK_M, BLOCK_K]
                x_offset = x_batch_off + spatial_off[:, None] + offs_k[None, :]
                x_mask = mask_sp[:, None] & mask_k[None, :]
                x_vals = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)

                # weights: w[oc, kh, kw, ic]
                w_offset = w_base + kw_base + offs_k[None, :]
                w_mask = mask_oc[:, None] & mask_k[None, :]
                w_vals = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)

                acc += tl.dot(x_vals, tl.trans(w_vals))

    # bias
    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + bias[None, :]

    # mish
    sp = tl.log(1.0 + tl.exp(acc))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    y = acc * th

    # bn fold
    scale = tl.load(scale_ptr + offs_oc, mask=mask_oc, other=0.0)
    shift = tl.load(shift_ptr + offs_oc, mask=mask_oc, other=0.0)
    out = y * scale[None, :] + shift[None, :]

    # Store to NHWC: [N, H_out, W_out, C_out] -- contiguous inner = C_out
    out_offset = (
        pid_n * (HW_out * C_out)
        + offs_sp[:, None] * C_out
        + offs_oc[None, :]
    )
    out_mask = mask_sp[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_offset, out, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps

        # Cache layout-converted weights for inference
        self._cached_w_nhwc = None
        self._cached_scale = None
        self._cached_shift = None
        self._cached_bias = None
        self._cache_valid = False

    def _build_cache(self, device, dtype):
        weight = self.conv.weight.detach().to(device=device, dtype=dtype)  # [OC, IC, KH, KW]
        # Permute to [OC, KH, KW, IC]
        w_nhwc = weight.permute(0, 2, 3, 1).contiguous()
        bias = (
            self.conv.bias.detach().to(device=device, dtype=dtype).contiguous()
            if self.conv.bias is not None
            else torch.zeros(self.out_channels, device=device, dtype=dtype)
        )
        running_mean = self.bn.running_mean.detach().to(device=device, dtype=dtype)
        running_var = self.bn.running_var.detach().to(device=device, dtype=dtype)
        bn_weight = self.bn.weight.detach().to(device=device, dtype=dtype)
        bn_bias = self.bn.bias.detach().to(device=device, dtype=dtype)
        invstd = torch.rsqrt(running_var + self.eps)
        scale = (bn_weight * invstd).contiguous()
        shift = (bn_bias - running_mean * scale).contiguous()

        self._cached_w_nhwc = w_nhwc
        self._cached_bias = bias
        self._cached_scale = scale
        self._cached_shift = shift
        self._cache_valid = True

    def forward(self, x):
        if self.training:
            x = self.conv(x)
            x = torch.multiply(torch.tanh(F.softplus(x)), x)
            x = self.bn(x)
            return x

        x = x.contiguous()
        N, C_in, H_in, W_in = x.shape
        C_out = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        H_out = H_in - KH + 1
        W_out = W_in - KW + 1

        if (not self._cache_valid) or (self._cached_w_nhwc.device != x.device) or (self._cached_w_nhwc.dtype != x.dtype):
            self._build_cache(x.device, x.dtype)

        # Convert input to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        out_nhwc = torch.empty((N, H_out, W_out, C_out), device=x.device, dtype=x.dtype)

        HW_out = H_out * W_out

        grid = lambda meta: (
            N,
            triton.cdiv(C_out, meta['BLOCK_N']),
            triton.cdiv(HW_out, meta['BLOCK_M']),
        )

        conv_mish_bn_nhwc_kernel[grid](
            x_nhwc, self._cached_w_nhwc, self._cached_bias,
            self._cached_scale, self._cached_shift,
            out_nhwc,
            N, C_in, H_in, W_in,
            C_out, KH, KW,
            H_out, W_out,
        )
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out