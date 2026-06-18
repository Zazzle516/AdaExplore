import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['IC', 'OC', 'IH', 'IW', 'KH', 'KW'],
)
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
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    sp_offs = pid_sp * BLOCK_M + tl.arange(0, BLOCK_M)
    oc_offs = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)

    oh = sp_offs // OW
    ow = sp_offs % OW
    sp_mask = sp_offs < (OH * OW)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K_total = IC * KH * KW

    for k_start in range(0, K_total, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K_total

        ic = k_offs // (KH * KW)
        rem = k_offs % (KH * KW)
        kh = rem // KW
        kw = rem % KW

        ih_num = oh[:, None] + PH - kh[None, :]
        iw_num = ow[:, None] + PW - kw[None, :]
        ih = ih_num // SH
        iw = iw_num // SW
        valid = ((ih_num % SH) == 0) & ((iw_num % SW) == 0) & \
                (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
        x_off = pid_n * (IC * IH * IW) + ic[None, :] * (IH * IW) + ih * IW + iw
        x_mask = valid & sp_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

        # weight pre-permuted to (IC, KH, KW, OC); contiguous along OC
        w_off = (ic[:, None] * (KH * KW * OC)
                 + kh[:, None] * (KW * OC)
                 + kw[:, None] * OC
                 + oc_offs[None, :])
        w_mask = k_mask[:, None] & oc_mask[None, :]
        w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

        acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b_vals[None, :]

    acc_t = tl.trans(acc)
    out_off = (pid_n * (OC * OH * OW)
               + oc_offs[:, None] * (OH * OW)
               + sp_offs[None, :])
    mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc_t, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 16}, num_warps=2),
        triton.Config({'BLOCK_H': 32}, num_warps=2),
        triton.Config({'BLOCK_H': 32}, num_warps=4),
        triton.Config({'BLOCK_H': 64}, num_warps=4),
        triton.Config({'BLOCK_H': 64}, num_warps=8),
        triton.Config({'BLOCK_H': 128}, num_warps=8),
    ],
    key=['OC', 'OH', 'OW'],
)
@triton.jit
def min_sum_gelu_bias_kernel(
    inp_ptr,
    bias_ptr,
    out_ptr,
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

        offs = (pid_n * OC * OH * OW
                + oc_range[:, None] * (OH * OW)
                + oh_offs[None, :] * OW
                + pid_w)
        mask = (oc_range[:, None] < OC) & h_mask[None, :]
        vals = tl.load(inp_ptr + offs, mask=mask, other=float('inf'))
        min_vals = tl.min(vals, axis=0)

        min_vals = tl.where(h_mask, min_vals, 0.0)
        sum_acc += tl.sum(min_vals, axis=0)

    x = sum_acc
    gelu = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865475))
    b = tl.load(bias_ptr)
    res = gelu + b

    out_off = pid_n * OW + pid_w
    tl.store(out_ptr + out_off, res)


def conv_transpose2d_triton(x, weight, bias_param, stride, padding, output_padding):
    N, IC, IH, IW = x.shape
    # weight is (IC, KH, KW, OC) - pre-permuted
    IC_w, KH, KW, OC = weight.shape
    assert IC == IC_w
    SH, SW = stride, stride
    PH, PW = padding, padding
    OH = (IH - 1) * SH - 2 * PH + KH + output_padding
    OW = (IW - 1) * SW - 2 * PW + KW + output_padding

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.float32)

    grid = lambda META: (N, triton.cdiv(OC, META['BLOCK_N']), triton.cdiv(OH * OW, META['BLOCK_M']))
    conv_transpose_gemm_kernel[grid](
        x, weight, bias_param, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW, SH, SW, PH, PW,
    )
    return out


def min_sum_gelu_bias_triton(inp, bias):
    N, OC, OH, OW = inp.shape
    out = torch.empty((N, 1, 1, OW), device=inp.device, dtype=torch.float32)
    # BLOCK_OC must be >= OC and a power of 2
    BLOCK_OC = 1
    while BLOCK_OC < OC:
        BLOCK_OC *= 2
    grid = (N, OW)
    min_sum_gelu_bias_kernel[grid](
        inp, bias, out,
        N, OC, OH, OW,
        BLOCK_OC=BLOCK_OC,
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
        self._w_perm_cache = None

    def _get_permuted_weight(self):
        # original weight shape: (IC, OC, KH, KW) -> (IC, KH, KW, OC)
        w = self.conv_transpose.weight
        return w.permute(0, 2, 3, 1).contiguous().cuda()

    def forward(self, x):
        x = x.contiguous().cuda()
        if self._w_perm_cache is None or self._w_perm_cache[0] is not self.conv_transpose.weight:
            w_perm = self._get_permuted_weight()
            self._w_perm_cache = (self.conv_transpose.weight, w_perm)
        else:
            w_perm = self._w_perm_cache[1]
        b = self.conv_transpose.bias.contiguous().cuda()
        conv_out = conv_transpose2d_triton(x, w_perm, b, self.stride, self.padding, self.output_padding)
        bias_scalar = self.bias.contiguous().view(-1)[0:1].cuda()
        out = min_sum_gelu_bias_triton(conv_out, bias_scalar)
        return out