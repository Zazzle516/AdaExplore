import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wic, stride_woc, stride_wd, stride_wh, stride_ww,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PADDING: tl.constexpr,
    MIN_VALUE: tl.constexpr, INV_DIVISOR: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    OHOW = OH * OW
    SP = OD * OH * OW

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < SP

    od = sp_offs // OHOW
    rem = sp_offs % OHOW
    oh = rem // OW
    ow = rem % OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # accumulator [BLOCK_OC, BLOCK_SP]
    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # For each kernel position, compute contributing input position
    for kd in tl.static_range(0, KD):
        id_num = od + PADDING - kd
        id_val = id_num // STRIDE
        id_valid = (id_num >= 0) & ((id_num % STRIDE) == 0) & (id_val >= 0) & (id_val < ID)
        for kh in tl.static_range(0, KH):
            ih_num = oh + PADDING - kh
            ih_val = ih_num // STRIDE
            ih_valid = (ih_num >= 0) & ((ih_num % STRIDE) == 0) & (ih_val >= 0) & (ih_val < IH)
            for kw in tl.static_range(0, KW):
                iw_num = ow + PADDING - kw
                iw_val = iw_num // STRIDE
                iw_valid = (iw_num >= 0) & ((iw_num % STRIDE) == 0) & (iw_val >= 0) & (iw_val < IW)

                sp_ok = id_valid & ih_valid & iw_valid & sp_mask  # [BLOCK_SP]

                # input offsets per output spatial pos: x[n, ic, id_val, ih_val, iw_val]
                in_sp_offset = (id_val * stride_xd + ih_val * stride_xh + iw_val * stride_xw)  # [BLOCK_SP]

                # Loop over IC
                for ic_start in range(0, IC, 16):
                    ic_offs = ic_start + tl.arange(0, 16)
                    ic_mask = ic_offs < IC

                    # Load x[n, ic_offs, ...] for each sp pos -> [16, BLOCK_SP]
                    x_ptrs = (x_ptr + pid_n * stride_xn
                              + ic_offs[:, None] * stride_xc
                              + in_sp_offset[None, :])
                    x_load_mask = ic_mask[:, None] & sp_ok[None, :]
                    x_vals = tl.load(x_ptrs, mask=x_load_mask, other=0.0)  # [16, BLOCK_SP]

                    # Load w[ic_offs, oc_offs, kd, kh, kw] -> [16, BLOCK_OC]
                    w_ptrs = (w_ptr + ic_offs[:, None] * stride_wic
                              + oc_offs[None, :] * stride_woc
                              + kd * stride_wd + kh * stride_wh + kw * stride_ww)
                    w_load_mask = ic_mask[:, None] & oc_mask[None, :]
                    w_vals = tl.load(w_ptrs, mask=w_load_mask, other=0.0)  # [16, BLOCK_OC]

                    # acc[oc, sp] += sum_ic w[ic, oc] * x[ic, sp]
                    acc += tl.dot(tl.trans(w_vals), x_vals)

    # Add bias, clamp, divide
    if b_ptr is not None:
        b = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
        acc = acc + b[:, None]

    acc = tl.where(acc < MIN_VALUE, MIN_VALUE, acc)
    acc = acc * INV_DIVISOR

    # store
    out_ptrs = (out_ptr + pid_n * stride_on
                + oc_offs[:, None] * stride_oc
                + sp_offs[None, :] * 1)  # ow stride is 1 if contiguous; use stride_ow=1
    # actually use proper strides:
    out_d = sp_offs // OHOW
    out_rem = sp_offs % OHOW
    out_h = out_rem // OW
    out_w = out_rem % OW
    out_sp_offset = out_d * stride_od + out_h * stride_oh + out_w * stride_ow
    out_ptrs = (out_ptr + pid_n * stride_on
                + oc_offs[:, None] * stride_oc
                + out_sp_offset[None, :])
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


def conv_transpose3d_fused(x, weight, bias, stride, padding, kernel_size, min_value, divisor):
    N, IC, ID, IH, IW = x.shape
    OC = weight.shape[1]
    KD = KH = KW = kernel_size
    OD = (ID - 1) * stride - 2 * padding + KD
    OH = (IH - 1) * stride - 2 * padding + KH
    OW = (IW - 1) * stride - 2 * padding + KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 32
    BLOCK_SP = 64

    SP = OD * OH * OW
    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(SP, BLOCK_SP))

    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        KD=KD, KH=KH, KW=KW,
        STRIDE=stride, PADDING=padding,
        MIN_VALUE=float(min_value), INV_DIVISOR=float(1.0 / divisor),
        BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
        num_warps=4, num_stages=2,
    )
    return out


@triton.jit
def fused_bias_clamp_div_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, channel_stride, n_channels,
    BLOCK_SIZE: tl.constexpr,
    MIN_VALUE: tl.constexpr,
    INV_DIVISOR: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = (offsets // channel_stride) % n_channels
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    x = x + b
    x = tl.where(x < MIN_VALUE, MIN_VALUE, x)
    x = x * INV_DIVISOR
    tl.store(out_ptr + offsets, x, mask=mask)


def fused_bias_clamp_div(x, bias, min_value, divisor):
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 4096
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    N, C, D, H, W = x.shape
    channel_stride = D * H * W
    fused_bias_clamp_div_kernel[grid](
        x, bias, out, n, channel_stride, C,
        BLOCK_SIZE=BLOCK_SIZE,
        MIN_VALUE=float(min_value),
        INV_DIVISOR=float(1.0 / divisor),
        num_warps=8,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super().__init__()
        conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.weight = nn.Parameter(conv.weight.detach().clone())
        self.bias = nn.Parameter(conv.bias.detach().clone())
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.min_value = min_value
        self.divisor = divisor

    def forward(self, x):
        x = x.contiguous()
        # Use cuDNN conv_transpose3d (without bias) then fuse epilogue
        out = F.conv_transpose3d(x, self.weight, None, stride=self.stride, padding=self.padding)
        out = fused_bias_clamp_div(out, self.bias, self.min_value, self.divisor)
        return out