import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# LayerNorm over last dim (which equals out_channels in this model's forward,
# because LayerNorm(out_channels) normalizes over the last tensor dim).
# Note: in the original model, x after convT is (N, C, D, H, W). LayerNorm
# with normalized_shape=out_channels normalizes over the LAST dim, which is W.
# But W is not OC. The reference torch.nn.LayerNorm requires the last dim's
# size to equal normalized_shape, otherwise it errors. Let's check: yes, torch
# would error if W != OC. With the given config OC=64 and W'=64 (after convT
# with stride=2, padding=1, kernel=4: 2*32 - 2*1 + 4 - 2 = 64... actually
# output formula: (W-1)*stride - 2*padding + kernel = 31*2 -2 +4 = 64). So
# yes W'=64 = OC. So LN is over the last dim of size 64.

@triton.jit
def _ln_gelu_scale_kernel(
    x_ptr, y_ptr, gamma_ptr, beta_ptr,
    N, C,
    eps, scale,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C

    x_row = x_ptr + row * C
    x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / C
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(gamma_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(beta_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    xn = xc * rstd
    y = xn * g + b

    inv_sqrt2 = 0.70710678118654752440
    y_gelu = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))
    out = y_gelu * scale

    tl.store(y_ptr + row * C + offs, out, mask=mask)


def ln_gelu_scale(x, gamma, beta, eps, scale):
    orig_shape = x.shape
    C = orig_shape[-1]
    x2d = x.reshape(-1, C).contiguous()
    N = x2d.shape[0]
    out = torch.empty_like(x2d)
    BLOCK_C = triton.next_power_of_2(C)
    num_warps = 4 if BLOCK_C <= 256 else 8
    _ln_gelu_scale_kernel[(N,)](
        x2d, out, gamma, beta,
        N, C, eps, scale,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
    )
    return out.reshape(orig_shape)


# ConvTranspose3d implemented as a gather-style matmul.
# For each output voxel (n, oc, od, oh, ow), accumulate over valid
# (ic, kd, kh, kw) such that:
#   od + pad_d - kd  ≡ 0 (mod stride_d), id = (od + pad_d - kd) / stride_d in [0, ID)
# similarly for oh, ow.
#
# We tile the output as (program_m = N * OD * OH * OW_tile, program_n = OC tile)
# and reduce over K = IC * KD * KH * KW.
#
# For efficiency we restructure: each program handles one (n, od, oh) and a
# tile over ow (BLOCK_OW) and a tile over oc (BLOCK_OC). The reduction over
# KD, KH, KW is small (4*4*4=64) and we loop IC in chunks of BLOCK_IC.

@triton.jit
def _convt3d_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wic, stride_woc, stride_wkd, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
    BLOCK_OC: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n_dh = tl.program_id(0)  # over N * OD * OH
    pid_ow = tl.program_id(1)    # over OW tiles
    pid_oc = tl.program_id(2)    # over OC tiles

    # decompose pid_n_dh
    n = pid_n_dh // (OD * OH)
    rem = pid_n_dh % (OD * OH)
    od = rem // OH
    oh = rem % OH

    ow_offs = pid_ow * BLOCK_OW + tl.arange(0, BLOCK_OW)  # [BLOCK_OW]
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    ow_mask = ow_offs < OW
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_OW, BLOCK_OC), dtype=tl.float32)

    # Precompute index components for od, oh dimensions
    # For kd in [0, KD): id = (od + PD - kd) / SD if divisible and in [0,ID)
    # We loop over kd, kh, kw explicitly (small).
    for kd in range(KD):
        id_num = od + PD - kd
        id_q = id_num // SD
        id_r = id_num - id_q * SD
        id_valid = (id_r == 0) & (id_q >= 0) & (id_q < ID)

        for kh in range(KH):
            ih_num = oh + PH - kh
            ih_q = ih_num // SH
            ih_r = ih_num - ih_q * SH
            ih_valid = (ih_r == 0) & (ih_q >= 0) & (ih_q < IH)

            dh_valid = id_valid & ih_valid

            for kw in range(KW):
                # vector over ow
                iw_num = ow_offs + PW - kw  # [BLOCK_OW]
                iw_q = iw_num // SW
                iw_r = iw_num - iw_q * SW
                iw_valid = (iw_r == 0) & (iw_q >= 0) & (iw_q < IW) & ow_mask

                spatial_valid = dh_valid & iw_valid  # [BLOCK_OW]

                # Loop over IC in chunks of BLOCK_IC
                for ic_start in range(0, IC, BLOCK_IC):
                    ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                    ic_mask = ic_offs < IC

                    # Load x[n, ic, id_q, ih_q, iw_q] -> [BLOCK_OW, BLOCK_IC]
                    # ptrs: x_ptr + n*stride_xn + ic*stride_xc + id*stride_xd + ih*stride_xh + iw*stride_xw
                    x_base = x_ptr + n * stride_xn + id_q * stride_xd + ih_q * stride_xh
                    x_ptrs = (x_base
                              + ic_offs[None, :] * stride_xc
                              + iw_q[:, None] * stride_xw)
                    x_load_mask = spatial_valid[:, None] & ic_mask[None, :]
                    x_vals = tl.load(x_ptrs, mask=x_load_mask, other=0.0)  # [BLOCK_OW, BLOCK_IC]

                    # Load w[ic, oc, kd, kh, kw] -> [BLOCK_IC, BLOCK_OC]
                    w_base = w_ptr + kd * stride_wkd + kh * stride_wkh + kw * stride_wkw
                    w_ptrs = (w_base
                              + ic_offs[:, None] * stride_wic
                              + oc_offs[None, :] * stride_woc)
                    w_load_mask = ic_mask[:, None] & oc_mask[None, :]
                    w_vals = tl.load(w_ptrs, mask=w_load_mask, other=0.0)  # [BLOCK_IC, BLOCK_OC]

                    acc += tl.dot(x_vals, w_vals, allow_tf32=True)

    # Add bias
    if b_ptr is not None:
        bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
        acc += bias[None, :]

    # Store
    y_base = y_ptr + n * stride_yn + od * stride_yd + oh * stride_yh
    y_ptrs = (y_base
              + oc_offs[None, :] * stride_yc
              + ow_offs[:, None] * stride_yw)
    store_mask = ow_mask[:, None] & oc_mask[None, :]
    tl.store(y_ptrs, acc, mask=store_mask)


def conv_transpose3d_triton(x, weight, bias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    SD, SH, SW = (stride, stride, stride) if isinstance(stride, int) else stride
    PD, PH, PW = (padding, padding, padding) if isinstance(padding, int) else padding

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    y = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    x = x.contiguous()
    w = weight.contiguous()
    if bias is not None:
        b = bias.contiguous()
    else:
        b = None

    BLOCK_OC = 32
    BLOCK_OW = 32
    BLOCK_IC = 16

    grid = (N * OD * OH, triton.cdiv(OW, BLOCK_OW), triton.cdiv(OC, BLOCK_OC))

    _convt3d_kernel[grid](
        x, w, b if b is not None else x,  # placeholder if no bias
        y,
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3), w.stride(4),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3), y.stride(4),
        BLOCK_OC=BLOCK_OC,
        BLOCK_OW=BLOCK_OW,
        BLOCK_IC=BLOCK_IC,
        num_warps=4,
        num_stages=2,
    )

    if b is None:
        # We passed x as a stand-in but the kernel branched on b_ptr is not None,
        # which is a Python-time check, so unreachable here. Recompute properly below.
        pass
    return y


# Because the kernel uses `if b_ptr is not None` at python compile time (the
# branch is baked in), we need two specialized variants. Simpler: always pass
# a real bias tensor (zeros if no bias).

@triton.jit
def _convt3d_kernel_bias(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wic, stride_woc, stride_wkd, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
    BLOCK_OC: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n_dh = tl.program_id(0)
    pid_ow = tl.program_id(1)
    pid_oc = tl.program_id(2)

    n = pid_n_dh // (OD * OH)
    rem = pid_n_dh % (OD * OH)
    od = rem // OH
    oh = rem % OH

    ow_offs = pid_ow * BLOCK_OW + tl.arange(0, BLOCK_OW)
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    ow_mask = ow_offs < OW
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_OW, BLOCK_OC), dtype=tl.float32)

    for kd in tl.static_range(KD):
        id_num = od + PD - kd
        id_q = id_num // SD
        id_r = id_num - id_q * SD
        id_valid = (id_r == 0) & (id_q >= 0) & (id_q < ID)

        for kh in tl.static_range(KH):
            ih_num = oh + PH - kh
            ih_q = ih_num // SH
            ih_r = ih_num - ih_q * SH
            ih_valid = (ih_r == 0) & (ih_q >= 0) & (ih_q < IH)

            dh_valid = id_valid & ih_valid

            for kw in tl.static_range(KW):
                iw_num = ow_offs + PW - kw
                iw_q = iw_num // SW
                iw_r = iw_num - iw_q * SW
                iw_valid = (iw_r == 0) & (iw_q >= 0) & (iw_q < IW) & ow_mask

                spatial_valid = dh_valid & iw_valid

                for ic_start in range(0, IC, BLOCK_IC):
                    ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                    ic_mask = ic_offs < IC

                    x_base = x_ptr + n * stride_xn + id_q * stride_xd + ih_q * stride_xh
                    x_ptrs = (x_base
                              + ic_offs[None, :] * stride_xc
                              + iw_q[:, None] * stride_xw)
                    x_load_mask = spatial_valid[:, None] & ic_mask[None, :]
                    x_vals = tl.load(x_ptrs, mask=x_load_mask, other=0.0)

                    w_base = w_ptr + kd * stride_wkd + kh * stride_wkh + kw * stride_wkw
                    w_ptrs = (w_base
                              + ic_offs[:, None] * stride_wic
                              + oc_offs[None, :] * stride_woc)
                    w_load_mask = ic_mask[:, None] & oc_mask[None, :]
                    w_vals = tl.load(w_ptrs, mask=w_load_mask, other=0.0)

                    acc += tl.dot(x_vals, w_vals, allow_tf32=True)

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += bias[None, :]

    y_base = y_ptr + n * stride_yn + od * stride_yd + oh * stride_yh
    y_ptrs = (y_base
              + oc_offs[None, :] * stride_yc
              + ow_offs[:, None] * stride_yw)
    store_mask = ow_mask[:, None] & oc_mask[None, :]
    tl.store(y_ptrs, acc, mask=store_mask)


def conv_transpose3d_triton_v2(x, weight, bias_tensor, stride, padding):
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    if isinstance(stride, int):
        SD = SH = SW = stride
    else:
        SD, SH, SW = stride
    if isinstance(padding, int):
        PD = PH = PW = padding
    else:
        PD, PH, PW = padding

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    y = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    x = x.contiguous()
    w = weight.contiguous()
    b = bias_tensor.contiguous()

    BLOCK_OC = 32
    BLOCK_OW = 32
    BLOCK_IC = 16

    grid = (N * OD * OH, triton.cdiv(OW, BLOCK_OW), triton.cdiv(OC, BLOCK_OC))

    _convt3d_kernel_bias[grid](
        x, w, b, y,
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3), w.stride(4),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3), y.stride(4),
        BLOCK_OC=BLOCK_OC,
        BLOCK_OW=BLOCK_OW,
        BLOCK_IC=BLOCK_IC,
        num_warps=4,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 bias=True, eps=1e-5, scaling_factor=1.0):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels,
                                                  kernel_size, stride=stride,
                                                  padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.eps = eps
        self.scaling_factor = float(scaling_factor)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.has_bias = bias

        # Pre-create a zero bias tensor for use if bias=False
        if not bias:
            self.register_buffer("_zero_bias", torch.zeros(out_channels))

    def forward(self, x):
        x = x.contiguous().cuda()
        if self.has_bias:
            bias_t = self.conv_transpose.bias
        else:
            bias_t = self._zero_bias.to(x.device, x.dtype)

        y = conv_transpose3d_triton_v2(
            x, self.conv_transpose.weight, bias_t,
            self.stride, self.padding,
        )

        out = ln_gelu_scale(y, self.layer_norm.weight, self.layer_norm.bias,
                            self.eps, self.scaling_factor)
        return out