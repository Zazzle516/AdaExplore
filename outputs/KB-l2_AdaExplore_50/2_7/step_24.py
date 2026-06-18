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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OS', 'K_TOT'],
)
@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, b_ptr, bias2_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    OS, K_TOT,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    IHW = IH * IW
    IDHW = ID * IHW
    KHW = KH * KW
    KDHW = KD * KHW
    x_base = pid_b * IC * IDHW

    ow = offs_m % OW
    oh = (offs_m // OW) % OH
    od = offs_m // (OW * OH)

    mask_m = offs_m < OS
    mask_n = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K_TOT, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_TOT

        kw = offs_k % KW
        kh = (offs_k // KW) % KH
        kd = (offs_k // KHW) % KD
        ic = offs_k // KDHW

        id_ = od[:, None] + kd[None, :]
        ih_ = oh[:, None] + kh[None, :]
        iw_ = ow[:, None] + kw[None, :]
        ic_ = ic[None, :]

        x_offset = (x_base
                    + ic_ * IDHW
                    + id_ * IHW
                    + ih_ * IW
                    + iw_)
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_vals = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)

        w_offset = (offs_n[None, :] * K_TOT + offs_k[:, None])
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_vals = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # Fused epilogue: ReLU -> LeakyReLU(0.01) -> GELU -> Sigmoid -> + bias2
    # For x >= 0, ReLU(x) = x, LeakyReLU(x) = x (identity)
    acc = tl.maximum(acc, 0.0)
    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))
    sig = 1.0 / (1.0 + tl.exp(-gelu))
    bias2 = tl.load(bias2_ptr + offs_n, mask=mask_n, other=0.0)
    out_val = sig + bias2[None, :]

    out_offset = (pid_b * OC * OS
                  + offs_n[None, :] * OS
                  + offs_m[:, None])
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offset, out_val, mask=out_mask)


def conv3d_triton_fused(x, weight, bias, bias2):
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    OS = OD * OH * OW
    K_TOT = IC * KD * KH * KW

    grid = lambda meta: (triton.cdiv(OS, meta['BLOCK_M']), N)

    conv3d_fused_kernel[grid](
        x, weight, bias, bias2, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        OS, K_TOT,
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
        bias_flat = self.bias.contiguous().cuda().view(-1)

        out = conv3d_triton_fused(x, weight, conv_bias, bias_flat)
        return out