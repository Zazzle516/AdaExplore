import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 256}, num_warps=8, num_stages=3),
    ],
    key=['IC', 'OC', 'OH', 'OW', 'KH', 'KW'],
)
@triton.jit
def conv_transpose_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH, KW, SH, SW, PH, PW,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oh = sp_offs // OW
    ow = sp_offs % OW

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (OH * OW)

    x_base = pid_n * (IC * IH * IW)
    IHW = IH * IW
    OCIC = OC * IC

    ic_offs = tl.arange(0, BLOCK_IC)
    ic_mask = ic_offs < IC

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    for kh_kw in range(0, KH * KW):
        kh = kh_kw // KW
        kw = kh_kw % KW

        ih_num = oh + PH - kh
        ih = ih_num // SH
        ih_valid = ((ih_num % SH) == 0) & (ih >= 0) & (ih < IH)
        iw_num = ow + PW - kw
        iw = iw_num // SW
        iw_valid = ((iw_num % SW) == 0) & (iw >= 0) & (iw < IW)
        valid = ih_valid & iw_valid & sp_mask

        x_sp_off = ih * IW + iw
        w_base = kh_kw * OCIC

        x_off = x_base + ic_offs[:, None] * IHW + x_sp_off[None, :]
        x_mask = ic_mask[:, None] & valid[None, :]
        x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

        w_off = w_base + oc_offs[:, None] * IC + ic_offs[None, :]
        w_mask = oc_mask[:, None] & ic_mask[None, :]
        w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

        acc += tl.dot(w_vals, x_vals, allow_tf32=True)

    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b_vals[:, None]

    out_off = pid_n * (OC * OH * OW) + oc_offs[:, None] * (OH * OW) + sp_offs[None, :]
    mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask)


@triton.jit
def min_sum_gelu_bias_kernel(
    inp_ptr,
    bias_ptr,
    out_ptr,
    N, OC, OH, OW,
    BLOCK_OC: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_wblk = tl.program_id(1)

    ow_offs = pid_wblk * BLOCK_OW + tl.arange(0, BLOCK_OW)
    ow_mask = ow_offs < OW

    oc_range = tl.arange(0, BLOCK_OC)

    sum_acc = tl.zeros((BLOCK_OW,), dtype=tl.float32)

    n_base = pid_n * OC * OH * OW

    for h in range(0, OH):
        # min over channels at this h, for each ow in tile
        min_vals = tl.full((BLOCK_OW,), float('inf'), dtype=tl.float32)
        for c_start in range(0, OC, BLOCK_OC):
            oc_offs = c_start + oc_range
            c_mask = oc_offs < OC
            offs = (n_base
                    + oc_offs[:, None] * (OH * OW)
                    + h * OW
                    + ow_offs[None, :])
            mask = c_mask[:, None] & ow_mask[None, :]
            vals = tl.load(inp_ptr + offs, mask=mask, other=float('inf'))
            tile_min = tl.min(vals, axis=0)
            min_vals = tl.minimum(min_vals, tile_min)
        sum_acc += min_vals

    x = sum_acc
    gelu = 0.5 * x * (1.0 + tl.erf(x / 1.4142135623730951))
    b = tl.load(bias_ptr)
    res = gelu + b

    out_off = pid_n * OW + ow_offs
    tl.store(out_ptr + out_off, res, mask=ow_mask)


def conv_transpose2d_triton(x, weight, bias_param, stride, padding, output_padding,
                            IC, OC, kernel_size):
    N, IC_x, IH, IW = x.shape
    KH = KW = kernel_size
    SH, SW = stride, stride
    PH, PW = padding, padding
    OH = (IH - 1) * SH - 2 * PH + KH + output_padding
    OW = (IW - 1) * SW - 2 * PW + KW + output_padding

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.float32)

    grid = lambda META: (N, triton.cdiv(OC, META['BLOCK_OC']), triton.cdiv(OH * OW, META['BLOCK_SP']))
    conv_transpose_kernel[grid](
        x, weight, bias_param, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW, SH, SW, PH, PW,
        BLOCK_IC=triton.next_power_of_2(IC),
    )
    return out


def min_sum_gelu_bias_triton(inp, bias):
    N, OC, OH, OW = inp.shape
    out = torch.empty((N, 1, 1, OW), device=inp.device, dtype=torch.float32)
    BLOCK_OC = 128
    BLOCK_OW = 128
    grid = (N, triton.cdiv(OW, BLOCK_OW))
    min_sum_gelu_bias_kernel[grid](
        inp, bias, out,
        N, OC, OH, OW,
        BLOCK_OC=BLOCK_OC, BLOCK_OW=BLOCK_OW,
        num_warps=8, num_stages=2,
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
        w = self.conv_transpose.weight
        w_perm = w.permute(2, 3, 1, 0).contiguous().cuda()
        return w_perm

    def forward(self, x):
        x = x.contiguous().cuda()
        if self._w_perm_cache is None or self._w_perm_cache[0] is not self.conv_transpose.weight:
            w_perm = self._get_permuted_weight()
            self._w_perm_cache = (self.conv_transpose.weight, w_perm)
        else:
            w_perm = self._w_perm_cache[1]
        b = self.conv_transpose.bias.contiguous().cuda()
        conv_out = conv_transpose2d_triton(x, w_perm, b, self.stride, self.padding,
                                           self.output_padding,
                                           self.in_channels, self.out_channels,
                                           self.kernel_size)
        bias_scalar = self.bias.contiguous().view(-1)[0:1].cuda()
        out = min_sum_gelu_bias_triton(conv_out, bias_scalar)
        return out