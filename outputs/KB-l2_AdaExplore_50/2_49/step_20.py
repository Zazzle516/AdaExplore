import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# ConvTranspose3d implemented as a scatter-add via input x weight outer product.
# One program handles a tile of (N*D_in*H_in*W_in) input positions x (OC tile).
# It loads input[n, :, d_in, h_in, w_in] (all IC), and for each (kd,kh,kw),
# computes weight^T @ input contribution and atomically adds to output.
#
# For correctness with overlapping output positions (stride>1 still has
# overlap from the kernel footprint), we use atomic_add.
# ---------------------------------------------------------------------------


@triton.jit
def conv_transpose3d_scatter_kernel(
    x_ptr,          # [N, IC, D_in, H_in, W_in]
    w_ptr,          # [IC, OC, KD, KH, KW]
    y_ptr,          # [N, OC, D_out, H_out, W_out]
    N, IC, OC,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    stride_d: tl.constexpr, stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_d: tl.constexpr, pad_h: tl.constexpr, pad_w: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # Each program: one input position (n, d_in, h_in, w_in), one OC tile.
    pid_pos = tl.program_id(0)        # over N * D_in * H_in * W_in
    pid_oc = tl.program_id(1)         # over OC tiles

    spatial = D_in * H_in * W_in
    n = pid_pos // spatial
    rem = pid_pos % spatial
    d_in = rem // (H_in * W_in)
    rem2 = rem % (H_in * W_in)
    h_in = rem2 // W_in
    w_in = rem2 % W_in

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    # Accumulate input vector across IC into a contribution per (kd,kh,kw, oc).
    # Strategy: load all IC for this single input position (IC <= ~128),
    # then loop over (kd,kh,kw) and compute dot(w[:, oc, kd, kh, kw], x[:]).

    offs_ic = tl.arange(0, BLOCK_IC)
    mask_ic = offs_ic < IC

    x_base = n * (IC * spatial) + d_in * (H_in * W_in) + h_in * W_in + w_in
    x_ptrs = x_ptr + x_base + offs_ic * spatial
    x_vec = tl.load(x_ptrs, mask=mask_ic, other=0.0)  # [BLOCK_IC]

    # weight layout: [IC, OC, KD, KH, KW]
    # stride for IC = OC*KD*KH*KW
    # stride for OC = KD*KH*KW
    KDKHKW = KD * KH * KW
    w_ic_stride = OC * KDKHKW
    w_oc_stride = KDKHKW

    for kd in tl.static_range(0, KD):
        d_out = d_in * stride_d - pad_d + kd
        d_valid = (d_out >= 0) & (d_out < D_out)
        for kh in tl.static_range(0, KH):
            h_out = h_in * stride_h - pad_h + kh
            h_valid = (h_out >= 0) & (h_out < H_out)
            for kw in tl.static_range(0, KW):
                w_out = w_in * stride_w - pad_w + kw
                w_valid = (w_out >= 0) & (w_out < W_out)
                valid = d_valid & h_valid & w_valid

                # Load weight slice [BLOCK_IC, BLOCK_OC] for this (kd,kh,kw).
                k_off = kd * (KH * KW) + kh * KW + kw
                w_ptrs = (w_ptr
                          + offs_ic[:, None] * w_ic_stride
                          + offs_oc[None, :] * w_oc_stride
                          + k_off)
                w_mask = mask_ic[:, None] & mask_oc[None, :]
                w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_IC, BLOCK_OC]

                # contrib[oc] = sum_ic x_vec[ic] * w_tile[ic, oc]
                contrib = tl.sum(x_vec[:, None] * w_tile, axis=0)  # [BLOCK_OC]

                if valid:
                    out_base = (n * (OC * D_out * H_out * W_out)
                                + d_out * (H_out * W_out)
                                + h_out * W_out
                                + w_out)
                    out_ptrs = y_ptr + out_base + offs_oc * (D_out * H_out * W_out)
                    tl.atomic_add(out_ptrs, contrib, mask=mask_oc)


def conv_transpose3d_triton(x, weight, bias, stride, padding, output_padding):
    """
    x: [N, IC, D_in, H_in, W_in]
    weight: [IC, OC, KD, KH, KW]
    bias: [OC] or None
    """
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()

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
        # initialize output to broadcasted bias (contiguous() materializes it)
        y = bias.view(1, OC, 1, 1, 1).expand(N, OC, D_out, H_out, W_out).contiguous()
    else:
        y = torch.zeros((N, OC, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_OC = 32
    BLOCK_IC = triton.next_power_of_2(IC)

    spatial = D_in * H_in * W_in
    grid = (N * spatial, triton.cdiv(OC, BLOCK_OC))

    conv_transpose3d_scatter_kernel[grid](
        x, weight, y,
        N, IC, OC,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        KD, KH, KW,
        sd, sh, sw,
        pd, ph, pw,
        BLOCK_OC=BLOCK_OC,
        BLOCK_IC=BLOCK_IC,
        num_warps=2,
        num_stages=2,
    )
    return y


# ---------------------------------------------------------------------------
# Fused softmax (over channel C) + sigmoid kernel.
# ---------------------------------------------------------------------------


@triton.jit
def softmax_sigmoid_kernel(
    x_ptr, out_ptr,
    N, C, S,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    num_s_blocks = tl.cdiv(S, BLOCK_S)
    n = pid // num_s_blocks
    s_blk = pid % num_s_blocks
    s_start = s_blk * BLOCK_S

    offs_c = tl.arange(0, BLOCK_C)
    offs_s = s_start + tl.arange(0, BLOCK_S)
    mask_c = offs_c < C
    mask_s = offs_s < S

    base = n * C * S
    # ptrs[c, s] = base + c*S + offs_s
    ptrs = x_ptr + base + offs_c[:, None] * S + offs_s[None, :]
    mask = mask_c[:, None] & mask_s[None, :]

    x = tl.load(ptrs, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=0)              # [BLOCK_S]
    e = tl.exp(x - m[None, :])
    e = tl.where(mask, e, 0.0)
    z = tl.sum(e, axis=0)              # [BLOCK_S]
    sm = e / z[None, :]
    out = 1.0 / (1.0 + tl.exp(-sm))

    out_ptrs = out_ptr + base + offs_c[:, None] * S + offs_s[None, :]
    tl.store(out_ptrs, out, mask=mask)


def fused_softmax_sigmoid(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    N, C = x.shape[0], x.shape[1]
    S = 1
    for d in x.shape[2:]:
        S *= d
    x_c = x.contiguous()
    out = torch.empty_like(x_c)

    BLOCK_C = triton.next_power_of_2(C)
    BLOCK_S = 32
    grid = (N * triton.cdiv(S, BLOCK_S),)
    softmax_sigmoid_kernel[grid](
        x_c, out, N, C, S,
        BLOCK_C=BLOCK_C,
        BLOCK_S=BLOCK_S,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super().__init__()
        # Hold a reference module so weight/bias initialization matches PyTorch.
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            output_padding=output_padding, bias=bias
        )

        # Normalize tuple parameters.
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

    def forward(self, x):
        x = x.contiguous()
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias
        y = conv_transpose3d_triton(
            x, weight, bias,
            self.stride, self.padding, self.output_padding
        )
        y = fused_softmax_sigmoid(y)
        return y