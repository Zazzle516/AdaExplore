import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    SUB: tl.constexpr,
    BLOCK_M: tl.constexpr,  # OC tile
    BLOCK_N: tl.constexpr,  # spatial tile
    BLOCK_K: tl.constexpr,  # K tile
    IC_CONST: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_sp = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # OC
    offs_n = pid_sp * BLOCK_N + tl.arange(0, BLOCK_N)  # spatial

    oh = offs_n // OW
    ow = offs_n % OW

    sp_mask = offs_n < (OH * OW)
    oc_mask = offs_m < OC

    K = IC_CONST * KH * KW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]
        k_mask = k_idx < K

        # decompose k_idx into (ic, kh, kw)
        ic = k_idx // (KH * KW)
        rem = k_idx % (KH * KW)
        kh = rem // KW
        kw = rem % KW

        # Load weight tile [BLOCK_M, BLOCK_K]
        w_off = offs_m[:, None] * K + k_idx[None, :]
        w_tile_mask = oc_mask[:, None] & k_mask[None, :]
        w_tile = tl.load(w_ptr + w_off, mask=w_tile_mask, other=0.0)

        # Load input tile [BLOCK_K, BLOCK_N]
        ih = oh[None, :] + kh[:, None]  # [BLOCK_K, BLOCK_N]
        iw = ow[None, :] + kw[:, None]

        x_off = (pid_n * (IC_CONST * IH * IW)
                 + ic[:, None] * (IH * IW)
                 + ih * IW
                 + iw)
        x_tile_mask = k_mask[:, None] & sp_mask[None, :]
        x_tile = tl.load(x_ptr + x_off, mask=x_tile_mask, other=0.0)

        acc += tl.dot(w_tile, x_tile)

    # bias
    b_vals = tl.load(b_ptr + offs_m, mask=oc_mask, other=0.0)
    acc = acc + b_vals[:, None] - SUB

    # mish
    sp = tl.where(acc > 20.0, acc, tl.log(1.0 + tl.exp(acc)))
    e2 = tl.exp(2.0 * sp)
    tanh_sp = (e2 - 1.0) / (e2 + 1.0)
    out = acc * tanh_sp

    out_off = (pid_n * (OC * OH * OW)
               + offs_m[:, None] * (OH * OW)
               + offs_n[None, :])
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, out, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC = w.shape[0]
        KH = w.shape[2]
        KW = w.shape[3]
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        SUB = float(self.subtract_value_1 + self.subtract_value_2)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 16  # IC=8, KH*KW=9 => K=72, use 16-wide

        grid = (N, triton.cdiv(OC, BLOCK_M), triton.cdiv(OH * OW, BLOCK_N))

        conv2d_mish_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            SUB,
            BLOCK_M, BLOCK_N, BLOCK_K,
            IC,
            num_warps=4, num_stages=3,
        )
        return out