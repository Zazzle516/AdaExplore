import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ConvTranspose3d as a true im2col-style GEMM with tl.dot
# Output[n, oc, od, oh, ow] = sum_{ic, kd, kh, kw} x[n, ic, id, ih, iw] * w[ic, oc, kd, kh, kw]
# where id = (od + PD - kd) / SD, etc., when divisible and in-range.
#
# We tile over (N, OC tile, output-spatial tile).
# K dimension iterates over (ic, kd, kh, kw) flattened.
# For each k step we gather x for all spatial outputs in the tile, building a
# [BLOCK_SP, BLOCK_K] tile, and load w as [BLOCK_K, BLOCK_OC], then tl.dot.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 32,  'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 256, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OD', 'OH', 'OW', 'IC', 'KD', 'KH', 'KW'],
)
@triton.jit
def conv_transpose3d_gemm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    K_TOTAL,  # IC*KD*KH*KW
    KDHW,     # KD*KH*KW
    KHW,      # KH*KW
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)         # [BLOCK_OC]
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)         # [BLOCK_SP]

    DHW = OD * OH * OW
    HW = OH * OW

    # decode spatial coords for each output position in tile
    od = sp_offs // HW
    rem = sp_offs % HW
    oh = rem // OW
    ow = rem % OW

    sp_mask = sp_offs < DHW                                       # [BLOCK_SP]
    oc_mask = oc_offs < OC                                        # [BLOCK_OC]

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    k_range = tl.arange(0, BLOCK_K)                               # [BLOCK_K]

    # x base for this n
    x_base_n = pid_n * IC * ID * IH * IW

    for k_start in range(0, K_TOTAL, BLOCK_K):
        k_idx = k_start + k_range                                 # [BLOCK_K]
        k_mask = k_idx < K_TOTAL

        # decompose k_idx into (ic, kd, kh, kw)
        ic_k = k_idx // KDHW
        krem = k_idx % KDHW
        kd_k = krem // KHW
        krem2 = krem % KHW
        kh_k = krem2 // KW
        kw_k = krem2 % KW

        # For each (sp, k): compute input position
        # id_num = od + PD - kd; valid if >=0, divisible by SD, id in [0,ID)
        id_num = od[:, None] + PD - kd_k[None, :]                 # [BLOCK_SP, BLOCK_K]
        ih_num = oh[:, None] + PH - kh_k[None, :]
        iw_num = ow[:, None] + PW - kw_k[None, :]

        id_v = id_num // SD
        ih_v = ih_num // SH
        iw_v = iw_num // SW

        valid = (
            (id_num >= 0) & ((id_num % SD) == 0) & (id_v >= 0) & (id_v < ID) &
            (ih_num >= 0) & ((ih_num % SH) == 0) & (ih_v >= 0) & (ih_v < IH) &
            (iw_num >= 0) & ((iw_num % SW) == 0) & (iw_v >= 0) & (iw_v < IW)
        )
        valid = valid & sp_mask[:, None] & k_mask[None, :]

        # x offset
        x_off = (x_base_n
                 + ic_k[None, :] * (ID * IH * IW)
                 + id_v * (IH * IW)
                 + ih_v * IW
                 + iw_v)
        x_tile = tl.load(x_ptr + x_off, mask=valid, other=0.0)    # [BLOCK_SP, BLOCK_K]

        # w offset: w[ic, oc, kd, kh, kw] => ic*OC*KDHW + oc*KDHW + kd*KHW + kh*KW + kw
        # We want [BLOCK_K, BLOCK_OC]
        w_off = (ic_k[:, None] * (OC * KDHW)
                 + oc_offs[None, :] * KDHW
                 + kd_k[:, None] * KHW
                 + kh_k[:, None] * KW
                 + kw_k[:, None])
        w_mask = k_mask[:, None] & oc_mask[None, :]
        w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)   # [BLOCK_K, BLOCK_OC]

        acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    # bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += bias[None, :]

    out_off = (pid_n * OC * DHW
               + oc_offs[None, :] * DHW
               + sp_offs[:, None])
    out_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


def conv_transpose3d_triton(x, w, b, stride, padding):
    N, IC, ID, IH, IW = x.shape
    IC2, OC, KD, KH, KW = w.shape
    assert IC == IC2
    SD = SH = SW = stride
    PD = PH = PW = padding

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

    K_TOTAL = IC * KD * KH * KW
    KDHW = KD * KH * KW
    KHW = KH * KW

    grid = lambda META: (
        N,
        triton.cdiv(OC, META['BLOCK_OC']),
        triton.cdiv(OD * OH * OW, META['BLOCK_SP']),
    )

    conv_transpose3d_gemm_kernel[grid](
        x, w, b, out,
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        K_TOTAL, KDHW, KHW,
    )
    return out


# Fused BN(train) + subtract spatial mean.
# In training mode, BN3d computes per-channel mean/var over (N,D,H,W).
# Final output simplifies (since shift cancels with spatial mean subtraction):
#   out = scale * (x - spatial_mean(x))   where scale = gamma / sqrt(var + eps)
# where var is computed over (N,D,H,W), but since BN's beta/mu are constant per
# (n,c) over spatial dims, they cancel against spatial mean.

@triton.jit
def fused_bn_submean_kernel(
    x_ptr, scale_ptr, out_ptr,
    N, C, SP,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    base = n * C * SP + c * SP

    s = tl.zeros((BLOCK,), dtype=tl.float32)
    for off in range(0, SP, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < SP
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        s += v
    mean = tl.sum(s) / SP

    scale = tl.load(scale_ptr + c)

    for off in range(0, SP, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < SP
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        out = scale * (v - mean)
        tl.store(out_ptr + base + idx, out, mask=mask)


def fused_bn_submean(x, scale):
    N, C, D, H, W = x.shape
    SP = D * H * W
    out = torch.empty_like(x)
    BLOCK = 1024
    grid = (N * C,)
    fused_bn_submean_kernel[grid](
        x, scale, out,
        N, C, SP,
        BLOCK=BLOCK, num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv_transpose.weight.contiguous()
        if self.conv_transpose.bias is not None:
            b = self.conv_transpose.bias.contiguous()
        else:
            b = torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)

        y = conv_transpose3d_triton(x, w, b, self.stride, self.padding)

        eps = self.batch_norm.eps
        dims = (0, 2, 3, 4)
        var = y.var(dim=dims, unbiased=False)
        gamma = self.batch_norm.weight
        scale = gamma / torch.sqrt(var + eps)

        out = fused_bn_submean(y, scale.contiguous())
        return out