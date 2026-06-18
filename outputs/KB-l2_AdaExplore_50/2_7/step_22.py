import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def fused_activation_bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, S,  # S = D*H*W
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # channel index
    c = (offsets // S) % C
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)

    # ReLU
    x = tl.maximum(x, 0.0)
    # LeakyReLU(0.01) - on non-negative this is identity
    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    # Sigmoid
    sig = 1.0 / (1.0 + tl.exp(-gelu))
    # Bias
    out = sig + b

    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    BLOCK_M: tl.constexpr,  # output spatial tile
    BLOCK_N: tl.constexpr,  # OC tile
    BLOCK_K: tl.constexpr,  # IC*KD*KH*KW reduction tile
):
    pid_m = tl.program_id(0)  # spatial output tile
    pid_n = tl.program_id(1)  # OC tile
    pid_b = tl.program_id(2)  # batch

    OS = OD * OH * OW  # spatial size
    K = IC * KD * KH * KW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # spatial positions
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # OC

    # decompose spatial position into (od, oh, ow)
    ow = offs_m % OW
    oh = (offs_m // OW) % OH
    od = offs_m // (OW * OH)

    mask_m = offs_m < OS
    mask_n = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # decompose k -> (ic, kd, kh, kw)
        kw = offs_k % KW
        kh = (offs_k // KW) % KH
        kd = (offs_k // (KW * KH)) % KD
        ic = offs_k // (KW * KH * KD)

        # input position
        # x[b, ic, od+kd, oh+kh, ow+kw]
        id_ = od[:, None] + kd[None, :]  # [BLOCK_M, BLOCK_K]
        ih_ = oh[:, None] + kh[None, :]
        iw_ = ow[:, None] + kw[None, :]
        ic_ = ic[None, :]

        x_offset = (pid_b * IC * ID * IH * IW
                    + ic_ * (ID * IH * IW)
                    + id_ * (IH * IW)
                    + ih_ * IW
                    + iw_)
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_vals = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # weight: w[oc, ic, kd, kh, kw], shape [OC, IC, KD, KH, KW]
        # we want w[offs_n, offs_k] -> [BLOCK_K, BLOCK_N]
        w_offset = (offs_n[None, :] * (IC * KD * KH * KW)
                    + offs_k[:, None])
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_vals = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        acc += tl.dot(x_vals, w_vals)

    # bias
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # store: out[b, oc, od, oh, ow]
    out_offset = (pid_b * OC * OS
                  + offs_n[None, :] * OS
                  + offs_m[:, None])
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offset, acc, mask=out_mask)


def conv3d_triton(x, weight, bias):
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_M = 64
    BLOCK_N = 32
    BLOCK_K = 16

    OS = OD * OH * OW
    grid = (triton.cdiv(OS, BLOCK_M), triton.cdiv(OC, BLOCK_N), N)

    conv3d_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv.weight.contiguous().cuda()
        conv_bias = self.conv.bias.contiguous().cuda()
        bias = self.bias.contiguous().cuda()

        y = conv3d_triton(x, weight, conv_bias)

        N, C, D, H, W = y.shape
        S = D * H * W
        total = y.numel()
        out = torch.empty_like(y)

        bias_flat = bias.view(-1).contiguous()

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(total, BLOCK_SIZE),)
        fused_activation_bias_kernel[grid](
            y, bias_flat, out,
            N, C, S, total,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4, num_stages=2,
        )
        return out