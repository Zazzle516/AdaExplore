import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K', 'KH', 'KW'],
)
@triton.jit
def conv_mish_bn_kernel(
    x_ptr,        # [N, C_in, H_in, W_in]
    w_ptr,        # [C_out, K] = [N, K] in GEMM terms (transposed access via offs_n[:, None]*K + k)
    b_ptr,        # [C_out]
    scale_ptr,    # [C_out]
    shift_ptr,    # [C_out]
    out_ptr,      # [N_batch, C_out, H_out, W_out]
    N_batch, C_in, H_in, W_in,
    C_out, KH, KW,
    H_out, W_out,
    M, N, K,
    HW_out,
    HW_in,
    KHW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # row in M = (n, oh, ow)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # oc

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Decode offs_m -> (n_b, oh, ow)
    n_b = offs_m // HW_out
    rem_m = offs_m % HW_out
    oh = rem_m // W_out
    ow = rem_m % W_out

    # x base offset per row
    x_row_base = n_b * (C_in * HW_in) + oh * W_in + ow  # [BLOCK_M]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]
        mask_k = k_idx < K

        ic = k_idx // KHW
        rem_k = k_idx % KHW
        kh = rem_k // KW
        kw = rem_k % KW

        # x offset = x_row_base[:, None] + ic*HW_in + kh*W_in + kw
        x_off = x_row_base[:, None] + ic[None, :] * HW_in + kh[None, :] * W_in + kw[None, :]
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

        # weight [C_out, K]: row-major. Need [K, BLOCK_N] for tl.dot
        # but easier: load as [BLOCK_N, BLOCK_K] then trans
        w_off = offs_n[:, None] * K + k_idx[None, :]
        w_mask = mask_n[:, None] & mask_k[None, :]
        w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, tl.trans(w_vals))

    # bias
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # mish: x * tanh(softplus(x))
    # softplus = log(1+exp(x)), tanh(sp) = (e^{2sp}-1)/(e^{2sp}+1)
    sp = tl.log(1.0 + tl.exp(acc))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    y = acc * th

    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)
    shift = tl.load(shift_ptr + offs_n, mask=mask_n, other=0.0)
    out = y * scale[None, :] + shift[None, :]

    # store to NCHW: out[n_b, oc, oh, ow]
    out_off = (
        n_b[:, None] * (C_out * HW_out)
        + offs_n[None, :] * HW_out
        + oh[:, None] * W_out
        + ow[:, None]
    )
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_off, out, mask=out_mask)


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
        N_batch, C_in, H_in, W_in = x.shape
        C_out = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        H_out = H_in - KH + 1
        W_out = W_in - KW + 1
        HW_out = H_out * W_out
        HW_in = H_in * W_in
        KHW = KH * KW
        K = C_in * KHW
        M = N_batch * HW_out
        N = C_out

        weight = self.conv.weight.contiguous().view(C_out, K)
        if self.conv.bias is not None:
            bias = self.conv.bias.contiguous()
        else:
            bias = torch.zeros(C_out, device=x.device, dtype=x.dtype)

        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        bn_weight = self.bn.weight
        bn_bias = self.bn.bias
        invstd = torch.rsqrt(running_var + self.eps)
        scale = (bn_weight * invstd).contiguous()
        shift = (bn_bias - running_mean * scale).contiguous()

        out = torch.empty((N_batch, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

        conv_mish_bn_kernel[grid](
            x, weight, bias,
            scale, shift,
            out,
            N_batch, C_in, H_in, W_in,
            C_out, KH, KW,
            H_out, W_out,
            M, N, K,
            HW_out,
            HW_in,
            KHW,
        )
        return out