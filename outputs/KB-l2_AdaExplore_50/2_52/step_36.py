import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    ],
    key=['N', 'C_in', 'H_in', 'W_in', 'C_out', 'KH', 'KW'],
)
@triton.jit
def conv_mish_bn_nhwc_kernel(
    x_ptr,        # NHWC: [N, H_in, W_in, C_in]
    w_ptr,        # [C_out, KH, KW, C_in]  (im2col-friendly: K = KH*KW*C_in)
    b_ptr,
    scale_ptr, shift_ptr,
    out_ptr,      # NHWC: [N, H_out, W_out, C_out]
    N, C_in, H_in, W_in,
    C_out, KH, KW,
    H_out, W_out,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)        # batch
    pid_oc = tl.program_id(1)       # out-channel tile
    pid_sp = tl.program_id(2)       # output-spatial tile

    HW_out = H_out * W_out

    offs_oc = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_sp = pid_sp * BLOCK_M + tl.arange(0, BLOCK_M)

    mask_oc = offs_oc < C_out
    mask_sp = offs_sp < HW_out

    oh = offs_sp // W_out
    ow = offs_sp % W_out

    K = KH * KW * C_in  # contiguous: kh, kw, ic

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # decompose k -> (kh, kw, ic)
        kh = offs_k // (KW * C_in)
        rem = offs_k % (KW * C_in)
        kw = rem // C_in
        ic = rem % C_in

        ih = oh[:, None] + kh[None, :]
        iw = ow[:, None] + kw[None, :]

        # NHWC layout: x[n, ih, iw, ic]
        x_offset = (
            pid_n * (H_in * W_in * C_in)
            + ih * (W_in * C_in)
            + iw * C_in
            + ic[None, :]
        )
        x_mask = mask_sp[:, None] & mask_k[None, :]
        x_vals = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)

        # weight layout [C_out, KH, KW, C_in] -> index [oc, k]
        w_offset = offs_oc[:, None] * K + offs_k[None, :]
        w_mask = mask_oc[:, None] & mask_k[None, :]
        w_vals = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, tl.trans(w_vals))

    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + bias[None, :]

    # mish: x * tanh(softplus(x))
    sp = tl.log(1.0 + tl.exp(acc))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    y = acc * th

    scale = tl.load(scale_ptr + offs_oc, mask=mask_oc, other=0.0)
    shift = tl.load(shift_ptr + offs_oc, mask=mask_oc, other=0.0)
    out = y * scale[None, :] + shift[None, :]

    # NHWC store: out[n, oh, ow, oc]
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

        self._cached = False
        self._w_nhwc = None
        self._bias = None
        self._scale = None
        self._shift = None

    def _build_cache(self, device, dtype):
        weight = self.conv.weight.detach().to(device=device, dtype=dtype)  # [Co, Ci, KH, KW]
        # Permute to [Co, KH, KW, Ci] contiguous
        w_nhwc = weight.permute(0, 2, 3, 1).contiguous()
        if self.conv.bias is not None:
            bias = self.conv.bias.detach().to(device=device, dtype=dtype).contiguous()
        else:
            bias = torch.zeros(self.out_channels, device=device, dtype=dtype)

        running_mean = self.bn.running_mean.detach().to(device=device, dtype=dtype)
        running_var = self.bn.running_var.detach().to(device=device, dtype=dtype)
        bn_weight = self.bn.weight.detach().to(device=device, dtype=dtype)
        bn_bias = self.bn.bias.detach().to(device=device, dtype=dtype)
        invstd = torch.rsqrt(running_var + self.eps)
        scale = (bn_weight * invstd).contiguous()
        shift = (bn_bias - running_mean * bn_weight * invstd).contiguous()

        self._w_nhwc = w_nhwc
        self._bias = bias
        self._scale = scale
        self._shift = shift
        self._cached = True

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

        if not self._cached:
            self._build_cache(x.device, x.dtype)

        # NHWC input
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        out_nhwc = torch.empty((N, H_out, W_out, C_out), device=x.device, dtype=x.dtype)

        HW_out = H_out * W_out

        grid = lambda meta: (
            N,
            triton.cdiv(C_out, meta['BLOCK_N']),
            triton.cdiv(HW_out, meta['BLOCK_M']),
        )

        conv_mish_bn_nhwc_kernel[grid](
            x_nhwc, self._w_nhwc, self._bias,
            self._scale, self._shift,
            out_nhwc,
            N, C_in, H_in, W_in,
            C_out, KH, KW,
            H_out, W_out,
        )

        # Convert back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out