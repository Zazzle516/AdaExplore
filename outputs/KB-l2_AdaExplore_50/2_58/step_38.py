import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_fused_kernel(
    x_ptr,           # (N, IC, D_in, H_in, W_in)
    w_ptr,           # (IC, OC, kD, kH, kW)
    cb_ptr,          # (OC,) conv bias
    bias_ptr,        # scalar bias (extra)
    out_ptr,         # (N, 1, D_out, H_out, W_out)
    N,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    IC: tl.constexpr,
    OC: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SD: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PD: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Program ids: (n, d_out, h_out * tile_w_idx)
    pid_nd = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_wt = tl.program_id(2)

    n = pid_nd // D_out
    d_out = pid_nd % D_out
    h_out = pid_h
    w_start = pid_wt * BLOCK_W

    w_offs = w_start + tl.arange(0, BLOCK_W)
    w_mask = w_offs < W_out

    # Accumulators for each OC channel (computed individually).
    # acc[oc, w] -> sum over (ic, kd, kh, kw)
    # We compute LSE across OC for each w in w_offs.

    # We'll iteratively compute each oc value and online-LSE reduce across oc.
    max_val = tl.full([BLOCK_W], -float('inf'), dtype=tl.float32)
    sum_exp = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Precompute valid kernel index ranges for d, h
    # d_in*SD = d_out + PD - kd  =>  kd = d_out + PD - d_in*SD
    # We loop over kd, kh, kw and compute corresponding d_in indices.

    DHW_in = D_in * H_in * W_in
    HW_in = H_in * W_in
    # weight layout (IC, OC, KD, KH, KW)
    W_OC_KDHW = OC * KD * KH * KW
    W_KDHW = KD * KH * KW
    W_KHW = KH * KW

    # Loop over OC channels; for each, compute conv-transpose value at (d_out, h_out, w_offs)
    for oc in tl.static_range(0, OC):
        acc = tl.zeros([BLOCK_W], dtype=tl.float32)

        for kd in tl.static_range(0, KD):
            d_in_num = d_out + PD - kd
            d_in = d_in_num // SD
            d_valid = (d_in_num >= 0) & (d_in_num % SD == 0) & (d_in < D_in) & (d_in >= 0)

            for kh in tl.static_range(0, KH):
                h_in_num = h_out + PH - kh
                h_in = h_in_num // SH
                h_valid = (h_in_num >= 0) & (h_in_num % SH == 0) & (h_in < H_in) & (h_in >= 0)

                dh_valid = d_valid & h_valid

                for kw in tl.static_range(0, KW):
                    w_in_num = w_offs + PW - kw
                    w_in = w_in_num // SW
                    w_valid = (w_in_num >= 0) & (w_in_num % SW == 0) & (w_in < W_in) & (w_in >= 0)
                    valid = w_valid & dh_valid & w_mask

                    # Sum over ic
                    for ic in tl.static_range(0, IC):
                        x_off = n * (IC * DHW_in) + ic * DHW_in + d_in * HW_in + h_in * W_in + w_in
                        w_off = ic * W_OC_KDHW + oc * W_KDHW + kd * W_KHW + kh * KW + kw
                        xv = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                        wv = tl.load(w_ptr + w_off)
                        acc += xv * wv

        # Add conv bias
        cb = tl.load(cb_ptr + oc)
        acc += cb

        # Online LSE update
        new_max = tl.maximum(max_val, acc)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.exp(acc - new_max)
        max_val = new_max

    lse = max_val + tl.log(sum_exp)

    # HardSwish: x * sigmoid(x+3) / 6
    hs = lse * tl.sigmoid(lse + 3.0) * (1.0 / 6.0)

    bias_val = tl.load(bias_ptr)
    res = hs - bias_val
    res = tl.minimum(tl.maximum(res, -1.0), 1.0)

    # Store: output shape (N, 1, D_out, H_out, W_out)
    out_off = n * (D_out * H_out * W_out) + d_out * (H_out * W_out) + h_out * W_out + w_offs
    tl.store(out_ptr + out_off, res, mask=w_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))

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

    def forward(self, x):
        x = x.contiguous()
        N, IC, D_in, H_in, W_in = x.shape
        KD, KH, KW = self.kernel_size
        SD, SH, SW = self.stride
        PD, PH, PW = self.padding

        D_out = (D_in - 1) * SD - 2 * PD + KD
        H_out = (H_in - 1) * SH - 2 * PH + KH
        W_out = (W_in - 1) * SW - 2 * PW + KW

        OC = self.out_channels

        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, KD, KH, KW)
        conv_bias = self.conv_transpose.bias
        if conv_bias is None:
            conv_bias = torch.zeros(OC, device=x.device, dtype=x.dtype)
        else:
            conv_bias = conv_bias.contiguous()

        bias_flat = self.bias.view(-1)[0:1]

        out = torch.empty((N, 1, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

        BLOCK_W = 32
        if W_out <= 32:
            BLOCK_W = 32
        elif W_out <= 64:
            BLOCK_W = 64
        else:
            BLOCK_W = 64

        grid = (N * D_out, H_out, (W_out + BLOCK_W - 1) // BLOCK_W)

        conv_transpose3d_fused_kernel[grid](
            x, weight, conv_bias, bias_flat, out,
            N,
            D_in, H_in, W_in,
            D_out, H_out, W_out,
            IC=IC,
            OC=OC,
            KD=KD, KH=KH, KW=KW,
            SD=SD, SH=SH, SW=SW,
            PD=PD, PH=PH, PW=PW,
            BLOCK_W=BLOCK_W,
            num_warps=4,
            num_stages=2,
        )
        return out