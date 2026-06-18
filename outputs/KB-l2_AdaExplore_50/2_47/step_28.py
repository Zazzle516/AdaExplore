import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OUT_SPATIAL', 'IC', 'KD', 'KH', 'KW'],
)
@triton.jit
def conv3d_mish_tanh_nhwc_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    OUT_SPATIAL,
    K_TOTAL: tl.constexpr,   # IC * KD * KH * KW
    BLOCK_M: tl.constexpr,   # spatial tile
    BLOCK_N: tl.constexpr,   # OC tile
    BLOCK_K: tl.constexpr,   # K reduction tile = IC (assumed power of 2)
):
    pid_n = tl.program_id(0)             # batch index
    pid_m = tl.program_id(1)             # spatial tile
    pid_oc = tl.program_id(2)            # OC tile

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)         # spatial linear idx
    offs_n = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)        # OC

    mask_m = offs_m < OUT_SPATIAL
    mask_n = offs_n < OC

    # decompose spatial idx -> (od, oh, ow)
    ow = offs_m % OW
    tmp = offs_m // OW
    oh = tmp % OH
    od = tmp // OH

    # base input coordinate for each output position
    id_base = od * SD - PD     # [BLOCK_M]
    ih_base = oh * SH - PH
    iw_base = ow * SW - PW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)  # = IC

    # NHWC layout: x stride is (IC*ID*IH*IW, 1, IC*IH*IW, IC*IW, IC)... we use NDHWC
    # x shape: (N, ID, IH, IW, IC); access x[n, id, ih, iw, ic]
    # offset = n*ID*IH*IW*IC + id*IH*IW*IC + ih*IW*IC + iw*IC + ic
    # weight layout: (KD, KH, KW, IC, OC) -- contiguous on OC
    # offset = kd*KH*KW*IC*OC + kh*KW*IC*OC + kw*IC*OC + ic*OC + oc

    x_n_off = pid_n * ID * IH * IW * IC

    for kd in tl.static_range(0, KD):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                id_ = id_base + kd
                ih_ = ih_base + kh
                iw_ = iw_base + kw
                in_bounds = (id_ >= 0) & (id_ < ID) & (ih_ >= 0) & (ih_ < IH) & (iw_ >= 0) & (iw_ < IW) & mask_m

                # x base offset per spatial (without IC)
                x_spatial_off = x_n_off + id_ * IH * IW * IC + ih_ * IW * IC + iw_ * IC  # [BLOCK_M]

                # x: [BLOCK_M, BLOCK_K]
                x_ptrs = x_ptr + x_spatial_off[:, None] + offs_k[None, :]
                x_mask = in_bounds[:, None] & (offs_k[None, :] < IC)
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

                # w: [BLOCK_K, BLOCK_N]
                w_kpos_off = kd * KH * KW * IC * OC + kh * KW * IC * OC + kw * IC * OC
                w_ptrs = w_ptr + w_kpos_off + offs_k[:, None] * OC + offs_n[None, :]
                w_mask = (offs_k[:, None] < IC) & mask_n[None, :]
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals, allow_tf32=True)

    # add bias
    b_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += b_vals[None, :]

    # fused mish + tanh
    sp = tl.log(1.0 + tl.exp(acc))
    t1 = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    m = acc * t1
    t2 = 2.0 * tl.sigmoid(2.0 * m) - 1.0

    # output NDHWC: (N, OD, OH, OW, OC)
    out_n_off = pid_n * OD * OH * OW * OC
    out_spatial_off = out_n_off + od * OH * OW * OC + oh * OW * OC + ow * OC  # [BLOCK_M]
    out_ptrs = out_ptr + out_spatial_off[:, None] + offs_n[None, :]
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, t2, mask=out_mask)


def conv3d_mish_tanh_nhwc(x_nhwc, w_nhwc, bias, stride, padding, N, IC, ID, IH, IW, OC, KD, KH, KW):
    SD, SH, SW = stride
    PD, PH, PW = padding
    OD = (ID + 2 * PD - KD) // SD + 1
    OH = (IH + 2 * PH - KH) // SH + 1
    OW = (IW + 2 * PW - KW) // SW + 1

    out = torch.empty((N, OD, OH, OW, OC), device=x_nhwc.device, dtype=x_nhwc.dtype)

    OUT_SPATIAL = OD * OH * OW

    # BLOCK_K must be >= IC and power-of-2 ideally. Use next pow2 of IC.
    BLOCK_K = 1
    while BLOCK_K < IC:
        BLOCK_K *= 2

    K_TOTAL = IC * KD * KH * KW

    grid = lambda META: (
        N,
        triton.cdiv(OUT_SPATIAL, META['BLOCK_M']),
        triton.cdiv(OC, META['BLOCK_N']),
    )

    conv3d_mish_tanh_nhwc_kernel[grid](
        x_nhwc, w_nhwc, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        OUT_SPATIAL,
        K_TOTAL,
        BLOCK_K=BLOCK_K,
    )
    return out, OD, OH, OW


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        if isinstance(stride, int):
            self.stride = (stride, stride, stride)
        else:
            self.stride = tuple(stride)
        if isinstance(padding, int):
            self.padding = (padding, padding, padding)
        else:
            self.padding = tuple(padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kd = self.kh = self.kw = kernel_size
        else:
            self.kd, self.kh, self.kw = kernel_size

        # Pre-permute weight to (KD, KH, KW, IC, OC) layout
        # original weight: (OC, IC, KD, KH, KW)
        with torch.no_grad():
            w = self.conv.weight.data
            w_perm = w.permute(2, 3, 4, 1, 0).contiguous()
        self.register_buffer('w_nhwc', w_perm, persistent=False)

    def _refresh_weight(self):
        # In case weights changed (e.g. training), refresh permuted weight
        with torch.no_grad():
            w = self.conv.weight.data
            w_perm = w.permute(2, 3, 4, 1, 0).contiguous()
            if self.w_nhwc.shape != w_perm.shape or self.w_nhwc.device != w_perm.device:
                self.w_nhwc = w_perm
            else:
                self.w_nhwc.copy_(w_perm)

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        # permute input to NDHWC
        x_nhwc = x.permute(0, 2, 3, 4, 1).contiguous()

        if self.training:
            self._refresh_weight()

        bias = self.conv.bias.contiguous()

        out_nhwc, OD, OH, OW = conv3d_mish_tanh_nhwc(
            x_nhwc, self.w_nhwc, bias,
            self.stride, self.padding,
            N, IC, ID, IH, IW, self.out_channels, self.kd, self.kh, self.kw,
        )

        # permute back to NCDHW
        out = out_nhwc.permute(0, 4, 1, 2, 3).contiguous()
        return out