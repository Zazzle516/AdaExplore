import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_scatter_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, OC,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    KD, KH, KW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # one program per (n, d_in, h_in, w_in, oc_tile)
    pid_oc = tl.program_id(0)
    pid_spatial = tl.program_id(1)
    pid_n = tl.program_id(2)

    n = pid_n
    w_in_idx = pid_spatial % W_in
    h_in_idx = (pid_spatial // W_in) % H_in
    d_in_idx = pid_spatial // (W_in * H_in)

    oc_start = pid_oc * BLOCK_OC
    offs_oc = oc_start + tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    # output base coords
    d_out_base = d_in_idx * stride_d - pad_d
    h_out_base = h_in_idx * stride_h - pad_h
    w_out_base = w_in_idx * stride_w - pad_w

    # accumulate input contributions: for each ic, read x[n,ic,d,h,w]
    # then for each (kd,kh,kw): out[n,oc, d_out_base+kd, h_out_base+kh, w_out_base+kw] += x*w[ic,oc,kd,kh,kw]
    # We'll loop over kd,kh,kw outer, and inside accumulate sum over IC for each oc tile.

    # x base for this (n, *, d_in, h_in, w_in)
    x_base = n * IC * D_in * H_in * W_in + d_in_idx * H_in * W_in + h_in_idx * W_in + w_in_idx
    # x stride along ic = D_in*H_in*W_in
    x_ic_stride = D_in * H_in * W_in

    # weight layout: [IC, OC, KD, KH, KW]
    w_oc_stride = KD * KH * KW
    w_ic_stride = OC * w_oc_stride

    for kd in range(0, KD):
        d_out = d_out_base + kd
        d_valid = (d_out >= 0) & (d_out < D_out)
        for kh in range(0, KH):
            h_out = h_out_base + kh
            h_valid = (h_out >= 0) & (h_out < H_out)
            for kw in range(0, KW):
                w_out = w_out_base + kw
                w_valid = (w_out >= 0) & (w_out < W_out)
                spatial_valid = d_valid & h_valid & w_valid

                # compute acc[BLOCK_OC] = sum over ic: x[ic] * w[ic, oc, kd, kh, kw]
                acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)
                for ic_start in range(0, IC, BLOCK_IC):
                    offs_ic = ic_start + tl.arange(0, BLOCK_IC)
                    mask_ic = offs_ic < IC
                    # load x[ic] : shape [BLOCK_IC]
                    x_ptrs = x_ptr + x_base + offs_ic * x_ic_stride
                    x_vals = tl.load(x_ptrs, mask=mask_ic, other=0.0)
                    # load w[ic, oc, kd, kh, kw] : shape [BLOCK_IC, BLOCK_OC]
                    w_ptrs = (w_ptr
                              + offs_ic[:, None] * w_ic_stride
                              + offs_oc[None, :] * w_oc_stride
                              + kd * KH * KW + kh * KW + kw)
                    w_mask = mask_ic[:, None] & mask_oc[None, :]
                    w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)
                    acc += tl.sum(x_vals[:, None] * w_vals, axis=0)

                # scatter-add into output
                if spatial_valid:
                    out_offset = (n * OC * D_out * H_out * W_out
                                  + offs_oc * D_out * H_out * W_out
                                  + d_out * H_out * W_out
                                  + h_out * W_out
                                  + w_out)
                    tl.atomic_add(out_ptr + out_offset, acc, mask=mask_oc)


def conv_transpose3d_triton(x, weight, bias, stride, padding, output_padding):
    N, IC, D_in, H_in, W_in = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    assert IC == IC_w

    sd, sh, sw = stride
    pd, ph, pw = padding
    opd, oph, opw = output_padding

    D_out = (D_in - 1) * sd - 2 * pd + KD + opd
    H_out = (H_in - 1) * sh - 2 * ph + KH + oph
    W_out = (W_in - 1) * sw - 2 * pw + KW + opw

    if bias is not None:
        out = bias.view(1, OC, 1, 1, 1).expand(N, OC, D_out, H_out, W_out).contiguous()
    else:
        out = torch.zeros((N, OC, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_OC = 64
    BLOCK_IC = 32

    grid = (
        triton.cdiv(OC, BLOCK_OC),
        D_in * H_in * W_in,
        N,
    )

    conv_transpose3d_scatter_kernel[grid](
        x, weight, bias if bias is not None else x, out,
        N, IC, OC,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        KD, KH, KW,
        sd, sh, sw,
        pd, ph, pw,
        BLOCK_OC=BLOCK_OC,
        BLOCK_IC=BLOCK_IC,
        num_warps=4,
        num_stages=2,
    )
    return out


@triton.jit
def softmax_sigmoid_kernel(
    x_ptr, out_ptr,
    N, C, S,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // S
    s = pid % S

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    base = n * C * S + s
    ptrs = x_ptr + base + offs_c * S

    x = tl.load(ptrs, mask=mask_c, other=-float('inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask_c, e, 0.0)
    z = tl.sum(e, axis=0)
    sm = e / z
    out = 1.0 / (1.0 + tl.exp(-sm))

    out_ptrs = out_ptr + base + offs_c * S
    tl.store(out_ptrs, out, mask=mask_c)


def fused_softmax_sigmoid(x: torch.Tensor) -> torch.Tensor:
    N, C = x.shape[0], x.shape[1]
    S = 1
    for d in x.shape[2:]:
        S *= d
    x_c = x.contiguous()
    out = torch.empty_like(x_c)

    BLOCK_C = triton.next_power_of_2(C)
    grid = (N * S,)
    softmax_sigmoid_kernel[grid](
        x_c, out, N, C, S,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        if isinstance(stride, int):
            self.stride = (stride, stride, stride)
        else:
            self.stride = tuple(stride)
        if isinstance(padding, int):
            self.padding = (padding, padding, padding)
        else:
            self.padding = tuple(padding)
        if isinstance(output_padding, int):
            self.output_padding = (output_padding, output_padding, output_padding)
        else:
            self.output_padding = tuple(output_padding)

        # match nn.ConvTranspose3d weight shape: [IC, OC, KD, KH, KW]
        ref = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            output_padding=output_padding, bias=bias
        )
        self.weight = nn.Parameter(ref.weight.detach().clone())
        if bias:
            self.bias = nn.Parameter(ref.bias.detach().clone())
        else:
            self.bias = None

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.weight.cuda()
        b = self.bias.cuda() if self.bias is not None else None
        x = conv_transpose3d_triton(x, w, b, self.stride, self.padding, self.output_padding)
        x = fused_softmax_sigmoid(x)
        return x