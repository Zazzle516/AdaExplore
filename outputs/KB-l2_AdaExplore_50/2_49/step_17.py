import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 32, 'BLOCK_OC': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 64, 'BLOCK_OC': 64, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 32, 'BLOCK_OC': 32, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 64, 'BLOCK_OC': 32, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 32, 'BLOCK_OC': 64, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 64, 'BLOCK_OC': 64, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 64, 'BLOCK_OC': 64, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 128, 'BLOCK_OC': 64, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'OC', 'OD', 'OH', 'OW', 'KD', 'KH', 'KW'],
)
@triton.jit
def conv_transpose3d_gather_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    HAS_BIAS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_oc = tl.program_id(1)
    n = tl.program_id(2)

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_ic = tl.arange(0, BLOCK_IC)

    OHW = OH * OW
    OS = OD * OHW

    s_mask = offs_s < OS
    oc_mask = offs_oc < OC

    od = offs_s // OHW
    rem = offs_s % OHW
    oh = rem // OW
    ow = rem % OW

    acc = tl.zeros((BLOCK_S, BLOCK_OC), dtype=tl.float32)

    IHW = IH * IW
    IDHW = ID * IHW
    x_n_base = n * IC * IDHW
    KDHW = KD * KH * KW
    KHW = KH * KW

    for kd in range(0, KD):
        d_num = od + PD - kd
        id_ = d_num // SD
        id_valid = (d_num >= 0) & (d_num - id_ * SD == 0) & (id_ >= 0) & (id_ < ID)

        for kh in range(0, KH):
            h_num = oh + PH - kh
            ih_ = h_num // SH
            ih_valid = (h_num >= 0) & (h_num - ih_ * SH == 0) & (ih_ >= 0) & (ih_ < IH)

            for kw in range(0, KW):
                w_num = ow + PW - kw
                iw_ = w_num // SW
                iw_valid = (w_num >= 0) & (w_num - iw_ * SW == 0) & (iw_ >= 0) & (iw_ < IW)

                spatial_valid = id_valid & ih_valid & iw_valid & s_mask
                in_spatial_off = id_ * IHW + ih_ * IW + iw_

                w_kernel_off = kd * KHW + kh * KW + kw

                for ic_start in range(0, IC, BLOCK_IC):
                    ic_idx = ic_start + offs_ic
                    ic_mask = ic_idx < IC

                    # x_tile: (BLOCK_S, BLOCK_IC)
                    x_off = x_n_base + ic_idx[None, :] * IDHW + in_spatial_off[:, None]
                    x_mask = spatial_valid[:, None] & ic_mask[None, :]
                    x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                    # w_tile: (BLOCK_IC, BLOCK_OC)
                    w_off = ic_idx[:, None] * (OC * KDHW) + offs_oc[None, :] * KDHW + w_kernel_off
                    w_mask = ic_mask[:, None] & oc_mask[None, :]
                    w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

                    acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    if HAS_BIAS:
        b_vals = tl.load(b_ptr + offs_oc, mask=oc_mask, other=0.0)
        acc += b_vals[None, :]

    out_off = n * (OC * OS) + offs_oc[None, :] * OS + offs_s[:, None]
    out_mask = s_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


def conv_transpose3d_triton(x, weight, bias, stride, padding, output_padding):
    N, IC, ID, IH, IW = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    assert IC == IC_w

    SD, SH, SW = stride if isinstance(stride, tuple) else (stride, stride, stride)
    PD, PH, PW = padding if isinstance(padding, tuple) else (padding, padding, padding)
    OPD, OPH, OPW = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding, output_padding)

    OD = (ID - 1) * SD - 2 * PD + KD + OPD
    OH = (IH - 1) * SH - 2 * PH + KH + OPH
    OW = (IW - 1) * SW - 2 * PW + KW + OPW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    OS = OD * OH * OW

    has_bias = bias is not None
    b_ptr = bias if has_bias else x  # dummy

    grid = lambda META: (triton.cdiv(OS, META['BLOCK_S']), triton.cdiv(OC, META['BLOCK_OC']), N)

    conv_transpose3d_gather_kernel[grid](
        x, weight, b_ptr, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        HAS_BIAS=has_bias,
    )
    return out


@triton.jit
def fused_softmax_sigmoid_kernel(
    x_ptr, out_ptr,
    N, C, S,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // S
    s = pid % S

    base = n * C * S + s
    offs_c = tl.arange(0, BLOCK_C)
    mask = offs_c < C
    ptrs = x_ptr + base + offs_c * S

    x = tl.load(ptrs, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s_sum = tl.sum(e, axis=0)
    sm = e / s_sum
    sig = 1.0 / (1.0 + tl.exp(-sm))
    tl.store(out_ptr + base + offs_c * S, sig, mask=mask)


def fused_softmax_sigmoid(x):
    N, C = x.shape[0], x.shape[1]
    spatial = x.shape[2] * x.shape[3] * x.shape[4]
    x_c = x.contiguous()
    out = torch.empty_like(x_c)

    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    if BLOCK_C <= 32:
        nw = 1
    elif BLOCK_C <= 64:
        nw = 1
    elif BLOCK_C <= 256:
        nw = 4
    else:
        nw = 8

    grid = (N * spatial,)
    fused_softmax_sigmoid_kernel[grid](
        x_c, out,
        N, C, spatial,
        BLOCK_C=BLOCK_C,
        num_warps=nw,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding, bias=bias
        )
        self.stride = self.conv_transpose.stride
        self.padding = self.conv_transpose.padding
        self.output_padding = self.conv_transpose.output_padding

    def forward(self, x):
        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()
        b = self.conv_transpose.bias
        if b is not None:
            b = b.contiguous()
        x = conv_transpose3d_triton(x, w, b, self.stride, self.padding, self.output_padding)
        x = fused_softmax_sigmoid(x)
        return x