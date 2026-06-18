import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_convtrans3d_lse_hs_bias_clamp_kernel(
    x_ptr,        # input: (N, IC, D_in, H_in, W_in)
    w_ptr,        # weight: (IC, OC, KD, KH, KW)
    b_ptr,        # conv bias: (OC,)
    bias_ptr,     # extra bias scalar
    out_ptr,      # (N, 1, D_out, H_out, W_out)
    N, IC, D_in, H_in, W_in,
    D_out, H_out, W_out,
    stride: tl.constexpr,
    padding: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    OC: tl.constexpr, IC_CONST: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # one program per (n, d_out, h_out, w_tile)
    pid = tl.program_id(0)
    pid_w = tl.program_id(1)

    DHW_blocks = D_out * H_out
    n = pid // DHW_blocks
    rem = pid % DHW_blocks
    d_out = rem // H_out
    h_out = rem % H_out

    w_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = w_offs < W_out

    bias_val = tl.load(bias_ptr)

    # accumulator: (BLOCK_W, OC)
    acc = tl.zeros([BLOCK_W, OC], dtype=tl.float32)
    # add conv bias (broadcast)
    oc_range = tl.arange(0, OC)
    b_vals = tl.load(b_ptr + oc_range)  # (OC,)
    acc = acc + b_vals[None, :]

    # output position relative offset
    # for ConvTranspose3d: out[n, oc, d_out, h_out, w_out] = sum over (ic, kd, kh, kw)
    #   x[n, ic, id, ih, iw] * w[ic, oc, kd, kh, kw]
    # where id*stride - padding + kd = d_out  =>  id = (d_out + padding - kd) / stride
    # need (d_out + padding - kd) % stride == 0  and 0 <= id < D_in

    DHW_in = D_in * H_in * W_in
    HW_in = H_in * W_in

    OC_KDHW = OC * KD * KH * KW
    KDHW = KD * KH * KW
    KHW = KH * KW

    for kd in tl.static_range(0, KD):
        d_num = d_out + padding - kd
        d_in = d_num // stride
        d_valid = ((d_num % stride) == 0) & (d_in >= 0) & (d_in < D_in)
        for kh in tl.static_range(0, KH):
            h_num = h_out + padding - kh
            h_in = h_num // stride
            h_valid = ((h_num % stride) == 0) & (h_in >= 0) & (h_in < H_in)
            for kw in tl.static_range(0, KW):
                w_num = w_offs + padding - kw
                w_in = w_num // stride
                w_valid = ((w_num % stride) == 0) & (w_in >= 0) & (w_in < W_in)
                spatial_valid = d_valid & h_valid & w_valid & w_mask

                # load input slice over all IC: shape (BLOCK_W, IC)
                # offset: n*IC*DHW_in + ic*DHW_in + d_in*HW_in + h_in*W_in + w_in
                base_in = n * IC_CONST * DHW_in + d_in * HW_in + h_in * W_in + w_in  # (BLOCK_W,)
                ic_range = tl.arange(0, IC_CONST)
                in_ptrs = base_in[:, None] + ic_range[None, :] * DHW_in  # (BLOCK_W, IC)
                in_mask = spatial_valid[:, None] & (ic_range[None, :] < IC_CONST)
                x_vals = tl.load(x_ptr + in_ptrs, mask=in_mask, other=0.0)  # (BLOCK_W, IC)

                # load weight slice: w[ic, oc, kd, kh, kw] -> shape (IC, OC)
                # offset: ic*OC*KDHW + oc*KDHW + kd*KHW + kh*KW + kw
                w_base = kd * KHW + kh * KW + kw
                w_ptrs = ic_range[:, None] * OC_KDHW + oc_range[None, :] * KDHW + w_base  # (IC, OC)
                w_vals = tl.load(w_ptr + w_ptrs)  # (IC, OC)

                # accumulate: (BLOCK_W, IC) @ (IC, OC) -> (BLOCK_W, OC)
                acc += tl.dot(x_vals, w_vals)

    # online LSE across OC dim (axis=1)
    max_val = tl.max(acc, axis=1)  # (BLOCK_W,)
    shifted = acc - max_val[:, None]
    sum_exp = tl.sum(tl.exp(shifted), axis=1)  # (BLOCK_W,)
    lse = max_val + tl.log(sum_exp)

    # hardswish
    hs = lse * tl.sigmoid(lse + 3.0) / 6.0
    res = hs - bias_val
    res = tl.minimum(tl.maximum(res, -1.0), 1.0)

    # store output: (N, 1, D_out, H_out, W_out)
    out_base = n * (D_out * H_out * W_out) + d_out * (H_out * W_out) + h_out * W_out + w_offs
    tl.store(out_ptr + out_base, res, mask=w_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        N, IC, D_in, H_in, W_in = x.shape
        KD = KH = KW = self.kernel_size
        stride = self.stride
        padding = self.padding
        D_out = (D_in - 1) * stride - 2 * padding + KD
        H_out = (H_in - 1) * stride - 2 * padding + KH
        W_out = (W_in - 1) * stride - 2 * padding + KW
        OC = self.out_channels

        x = x.contiguous()
        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, KD, KH, KW)
        conv_bias = self.conv_transpose.bias.contiguous()
        bias_flat = self.bias.view(-1)[0:1]

        out = torch.empty((N, 1, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

        BLOCK_W = 64 if W_out >= 64 else triton.next_power_of_2(W_out)
        if BLOCK_W < 16:
            BLOCK_W = 16
        grid = (N * D_out * H_out, (W_out + BLOCK_W - 1) // BLOCK_W)

        fused_convtrans3d_lse_hs_bias_clamp_kernel[grid](
            x, weight, conv_bias, bias_flat, out,
            N, IC, D_in, H_in, W_in,
            D_out, H_out, W_out,
            stride=stride, padding=padding,
            KD=KD, KH=KH, KW=KW,
            OC=OC, IC_CONST=IC,
            BLOCK_W=BLOCK_W,
            num_warps=4, num_stages=2,
        )
        return out