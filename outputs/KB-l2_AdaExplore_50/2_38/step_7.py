import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    inp_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, od, oh, ow_tile), iterate OC and IC
    pid = tl.program_id(0)
    pid_ow = tl.program_id(1)
    # decode pid
    ow_tile = pid_ow  # not used as tile, single ow
    # pid encodes (n, od, oh)
    n = pid // (OD * OH)
    rem = pid % (OD * OH)
    od = rem // OH
    oh = rem % OH
    ow = pid_ow

    # output (n, :, od, oh, ow)
    oc_offs = tl.arange(0, BLOCK_OC)
    
    # bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_offs < OC, other=0.0)
    acc = bias

    # For ConvTranspose3d:
    # output[n,oc,od,oh,ow] = sum over ic, kd, kh, kw of:
    #   input[n, ic, id, ih, iw] * weight[ic, oc, kd, kh, kw]
    # where: od = id*stride - pad + kd  =>  id = (od + pad - kd) / stride
    # valid if (od+pad-kd) divisible by stride and 0 <= id < ID
    
    for kd in tl.static_range(0, KD):
        id_num = od + PAD - kd
        id_q = id_num // STRIDE
        id_r = id_num - id_q * STRIDE
        valid_d = (id_r == 0) & (id_q >= 0) & (id_q < ID)
        for kh in tl.static_range(0, KH):
            ih_num = oh + PAD - kh
            ih_q = ih_num // STRIDE
            ih_r = ih_num - ih_q * STRIDE
            valid_h = (ih_r == 0) & (ih_q >= 0) & (ih_q < IH)
            for kw in tl.static_range(0, KW):
                iw_num = ow + PAD - kw
                iw_q = iw_num // STRIDE
                iw_r = iw_num - iw_q * STRIDE
                valid_w = (iw_r == 0) & (iw_q >= 0) & (iw_q < IW)
                valid = valid_d & valid_h & valid_w

                if valid:
                    # Load input vector [IC] at (n, :, id_q, ih_q, iw_q)
                    # and weight matrix [IC, OC] at (:, :, kd, kh, kw)
                    # Loop over IC
                    for ic in range(0, IC):
                        in_off = ((n * IC + ic) * ID + id_q) * IH * IW + ih_q * IW + iw_q
                        x_val = tl.load(inp_ptr + in_off)
                        # weight shape: (IC, OC, KD, KH, KW)
                        w_off = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                        w_val = tl.load(w_ptr + w_off, mask=oc_offs < OC, other=0.0)
                        acc += x_val * w_val

    # store
    out_off = ((n * OC + oc_offs) * OD + od) * OH * OW + oh * OW + ow
    tl.store(out_ptr + out_off, acc, mask=oc_offs < OC)


@triton.jit
def softmax_scale_kernel(
    x_ptr, scale_ptr, out_ptr,
    S,
    clamp_min: tl.constexpr, clamp_max: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid % tl.num_programs(0)  # not used
    # we passed grid = (B*C,), pid is row index
    row = pid
    # determine c from row: row = b*C + c; we need scale[c]
    # pass C via specialization isn't needed; use modulo with C
    pass


@triton.jit
def softmax_scale_kernel2(
    x_ptr, scale_ptr, out_ptr,
    C, S,
    clamp_min: tl.constexpr, clamp_max: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid % C
    row_start = pid * S
    scale = tl.load(scale_ptr + c)

    max_val = -float('inf')
    sum_exp = 0.0
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=-float('inf'))
        v = tl.minimum(tl.maximum(v, clamp_min), clamp_max)
        cur_max = tl.max(v, axis=0)
        new_max = tl.maximum(max_val, cur_max)
        e = tl.exp(v - new_max)
        e = tl.where(mask, e, 0.0)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(e, axis=0)
        max_val = new_max

    inv = 1.0 / sum_exp

    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        v = tl.minimum(tl.maximum(v, clamp_min), clamp_max)
        e = tl.exp(v - max_val) * inv * scale
        tl.store(out_ptr + row_start + idx, e, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super().__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        b, c, d, h, w = x.shape
        S = d * h * w
        x_flat = x.contiguous().view(b * c, S)
        out = torch.empty_like(x_flat)
        scale_flat = self.scale.view(-1).contiguous()

        if S >= 2048:
            BLOCK = 2048
            num_warps = 8
        elif S >= 1024:
            BLOCK = 1024
            num_warps = 8
        elif S >= 512:
            BLOCK = 512
            num_warps = 4
        else:
            BLOCK = 256
            num_warps = 4

        grid = (b * c,)
        softmax_scale_kernel2[grid](
            x_flat, scale_flat, out,
            c, S,
            self.clamp_min, self.clamp_max,
            BLOCK=BLOCK, num_warps=num_warps, num_stages=2,
        )
        return out.view(b, c, d, h, w)