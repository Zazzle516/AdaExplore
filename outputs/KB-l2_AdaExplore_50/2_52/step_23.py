import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['N', 'C_in', 'H_in', 'W_in', 'C_out', 'KH', 'KW'],
)
@triton.jit
def conv_mish_bn_kernel(
    x_ptr, w_ptr, b_ptr,
    scale_ptr, shift_ptr,
    out_ptr,
    N, C_in, H_in, W_in,
    C_out, KH, KW,
    H_out, W_out,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)  # batch index
    pid_oc = tl.program_id(1)  # block over out channels
    pid_sp = tl.program_id(2)  # block over output spatial

    HW_out = H_out * W_out

    offs_oc = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_sp = pid_sp * BLOCK_M + tl.arange(0, BLOCK_M)

    mask_oc = offs_oc < C_out
    mask_sp = offs_sp < HW_out

    oh = offs_sp // W_out
    ow = offs_sp % W_out

    K = C_in * KH * KW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # decompose k -> (ic, kh, kw)
        ic = offs_k // (KH * KW)
        rem = offs_k % (KH * KW)
        kh = rem // KW
        kw = rem % KW

        # input positions for each (m, k)
        ih = oh[:, None] + kh[None, :]  # [BLOCK_M, BLOCK_K]
        iw = ow[:, None] + kw[None, :]

        x_offset = (
            pid_n * (C_in * H_in * W_in)
            + ic[None, :] * (H_in * W_in)
            + ih * W_in
            + iw
        )
        x_mask = mask_sp[:, None] & mask_k[None, :]
        x_vals = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)

        # weight [C_out, C_in, KH, KW] -> index [oc, k]
        w_offset = offs_oc[:, None] * K + offs_k[None, :]
        w_mask = mask_oc[:, None] & mask_k[None, :]
        w_vals = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)

        # acc += x_vals @ w_vals.T
        acc += tl.dot(x_vals, tl.trans(w_vals))

    # bias
    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + bias[None, :]

    # mish: x * tanh(softplus(x))
    # tanh(softplus(x)) = 1 - 2/(e^(2*softplus(x))+1) = 1 - 2/((1+e^x)^2+1)
    ex = tl.exp(acc)
    one_plus_ex = 1.0 + ex
    denom = one_plus_ex * one_plus_ex + 1.0
    th = 1.0 - 2.0 / denom
    y = acc * th

    # batchnorm: scale * y + shift
    scale = tl.load(scale_ptr + offs_oc, mask=mask_oc, other=0.0)
    shift = tl.load(shift_ptr + offs_oc, mask=mask_oc, other=0.0)
    out = y * scale[None, :] + shift[None, :]

    # store transposed: [BLOCK_N, BLOCK_M] so spatial (contiguous) is the last axis
    out_t = tl.trans(out)
    out_offset = (
        pid_n * (C_out * HW_out)
        + offs_oc[:, None] * HW_out
        + offs_sp[None, :]
    )
    out_mask = mask_oc[:, None] & mask_sp[None, :]
    tl.store(out_ptr + out_offset, out_t, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps

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

        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous() if self.conv.bias is not None else torch.zeros(C_out, device=x.device, dtype=x.dtype)

        # BN folded params
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        bn_weight = self.bn.weight
        bn_bias = self.bn.bias
        invstd = torch.rsqrt(running_var + self.eps)
        scale = bn_weight * invstd
        shift = bn_bias - running_mean * scale
        scale = scale.contiguous()
        shift = shift.contiguous()

        out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

        HW_out = H_out * W_out

        grid = lambda meta: (
            N,
            triton.cdiv(C_out, meta['BLOCK_N']),
            triton.cdiv(HW_out, meta['BLOCK_M']),
        )

        conv_mish_bn_kernel[grid](
            x, weight, bias,
            scale, shift,
            out,
            N, C_in, H_in, W_in,
            C_out, KH, KW,
            H_out, W_out,
        )
        return out