import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_gemm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    For each (n, oc_tile, sp_tile), compute conv_transpose output via GEMM:
      out[n, oc, oh, ow] = sum_{ic, kh, kw} x[n, ic, ih, iw] * W[ic, oc, kh, kw]
      where ih = (oh + PH - kh) / SH (only valid when divisible & in range)
    
    GEMM dims:
      M = BLOCK_M output spatial positions
      N = BLOCK_N output channels  
      K = IC * KH * KW (reduction)
    """
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    sp_offs = pid_sp * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    oc_offs = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    oh = sp_offs // OW
    ow = sp_offs % OW
    sp_mask = sp_offs < (OH * OW)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K_total = IC * KH * KW

    # iterate over K in blocks
    for k_start in range(0, K_total, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_offs < K_total

        ic = k_offs // (KH * KW)
        rem = k_offs % (KH * KW)
        kh = rem // KW
        kw = rem % KW

        # gather x: shape [BLOCK_M, BLOCK_K]
        ih_num = oh[:, None] + PH - kh[None, :]
        iw_num = ow[:, None] + PW - kw[None, :]
        ih = ih_num // SH
        iw = iw_num // SW
        valid = ((ih_num % SH) == 0) & ((iw_num % SW) == 0) & \
                (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
        x_off = pid_n * (IC * IH * IW) + ic[None, :] * (IH * IW) + ih * IW + iw
        x_mask = valid & sp_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # gather w: shape [BLOCK_K, BLOCK_N]
        # W is (IC, OC, KH, KW) contiguous; index [ic, oc, kh, kw]
        w_off = (ic[:, None] * (OC * KH * KW)
                 + oc_offs[None, :] * (KH * KW)
                 + kh[:, None] * KW
                 + kw[:, None])
        w_mask = k_mask[:, None] & oc_mask[None, :]
        w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    # bias
    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b_vals[None, :]

    # store: out is (N, OC, OH, OW) contiguous. We need [BLOCK_M sp, BLOCK_N oc].
    # offset = n*OC*OH*OW + oc*(OH*OW) + sp
    out_off = (pid_n * (OC * OH * OW)
               + oc_offs[None, :] * (OH * OW)
               + sp_offs[:, None])
    mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask)


@triton.jit
def min_sum_gelu_bias_kernel(
    inp_ptr,   # (N, OC, OH, OW)
    bias_ptr,  # (1,)
    out_ptr,   # (N, OW)
    N, OC, OH, OW,
    BLOCK_H: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_w = tl.program_id(1)

    oc_range = tl.arange(0, BLOCK_OC)
    oh_range = tl.arange(0, BLOCK_H)

    sum_acc = 0.0

    for h_start in range(0, OH, BLOCK_H):
        oh_offs = h_start + oh_range
        h_mask = oh_offs < OH

        min_vals = tl.full((BLOCK_H,), float('inf'), dtype=tl.float32)
        for c_start in range(0, OC, BLOCK_OC):
            oc_offs = c_start + oc_range
            c_mask = oc_offs < OC
            offs = (pid_n * OC * OH * OW
                    + oc_offs[:, None] * (OH * OW)
                    + oh_offs[None, :] * OW
                    + pid_w)
            mask = c_mask[:, None] & h_mask[None, :]
            vals = tl.load(inp_ptr + offs, mask=mask, other=float('inf'))
            tile_min = tl.min(vals, axis=0)
            min_vals = tl.minimum(min_vals, tile_min)

        min_vals = tl.where(h_mask, min_vals, 0.0)
        sum_acc += tl.sum(min_vals, axis=0)

    x = sum_acc
    gelu = 0.5 * x * (1.0 + tl.erf(x / 1.4142135623730951))
    b = tl.load(bias_ptr)
    res = gelu + b

    out_off = pid_n * OW + pid_w
    tl.store(out_ptr + out_off, res)


def conv_transpose2d_triton(x, weight, bias_param, stride, padding, output_padding):
    N, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w
    SH, SW = stride, stride
    PH, PW = padding, padding
    OH = (IH - 1) * SH - 2 * PH + KH + output_padding
    OW = (IW - 1) * SW - 2 * PW + KW + output_padding

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (N, triton.cdiv(OC, BLOCK_N), triton.cdiv(OH * OW, BLOCK_M))
    conv_transpose_gemm_kernel[grid](
        x, weight, bias_param, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW, SH, SW, PH, PW,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return out


def min_sum_gelu_bias_triton(inp, bias):
    N, OC, OH, OW = inp.shape
    out = torch.empty((N, 1, 1, OW), device=inp.device, dtype=torch.float32)
    BLOCK_OC = 128
    BLOCK_H = 32
    grid = (N, OW)
    min_sum_gelu_bias_kernel[grid](
        inp, bias, out,
        N, OC, OH, OW,
        BLOCK_H=BLOCK_H, BLOCK_OC=BLOCK_OC,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv_transpose.weight.contiguous().cuda()
        b = self.conv_transpose.bias.contiguous().cuda()
        conv_out = conv_transpose2d_triton(x, w, b, self.stride, self.padding, self.output_padding)
        bias_scalar = self.bias.contiguous().view(-1)[0:1].cuda()
        out = min_sum_gelu_bias_triton(conv_out, bias_scalar)
        return out