import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Scatter-add ConvTranspose3d kernel:
# One program per (n, ic_tile, id, ih, iw_tile)
# For each input element, compute its contribution to OC output positions
# and atomically add into the output.
# But to reduce atomics, we do: one program per (n, od_tile?, ...) gather form is better.
#
# We'll use the gather form: one program per (n, oc_tile, od, oh, ow_tile),
# with an inner reduction loop over (ic, kd, kh, kw). This is the standard im2col approach.
# We tile over OC and OW. The inner loop over IC*KD*KH*KW does tl.dot.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_OW': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_OW': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_OW': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OW', 'IC', 'KD', 'KH', 'KW'],
)
@triton.jit
def conv_transpose3d_gemm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_OW: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (n * OD * OH, num_oc_tiles, num_ow_tiles)
    pid_ndh = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_ow = tl.program_id(2)

    n = pid_ndh // (OD * OH)
    rem = pid_ndh % (OD * OH)
    od = rem // OH
    oh = rem % OH

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BM]
    ow_offs = pid_ow * BLOCK_OW + tl.arange(0, BLOCK_OW)  # [BN]

    oc_mask = oc_offs < OC  # [BM]
    ow_mask = ow_offs < OW  # [BN]

    # accumulator [BM, BN]
    acc = tl.zeros([BLOCK_OC, BLOCK_OW], dtype=tl.float32)

    # K dimension is IC * KD * KH * KW
    K_total = IC * KD * KH * KW

    # Precompute per-ow validity for each kw
    # For each (kd, kh, kw): id_num = od + PD - kd, ih_num = oh + PH - kh, iw_num[ow] = ow + PW - kw
    # id_ valid same across BN; iw_ varies with ow.
    # We loop over (kd, kh, kw) outer, inner-loop over ic in chunks of BLOCK_K? No, BLOCK_K spans ic.
    # Strategy: outermost loop over kd, kh, kw; middle loop over ic in chunks of BLOCK_K; do tl.dot.

    # weight layout: [IC, OC, KD, KH, KW], w_idx = ((ic*OC+oc)*KD+kd)*KH*KW + kh*KW + kw

    for kd in tl.static_range(0, KD):
        id_num = od + PD - kd
        id_ = id_num // SD
        id_valid = ((id_num - id_ * SD) == 0) & (id_ >= 0) & (id_ < ID)
        for kh in tl.static_range(0, KH):
            ih_num = oh + PH - kh
            ih = ih_num // SH
            ih_valid = ((ih_num - ih * SH) == 0) & (ih >= 0) & (ih < IH)
            for kw in tl.static_range(0, KW):
                iw_num = ow_offs + PW - kw  # [BN]
                iw = iw_num // SW           # [BN]
                iw_valid = ((iw_num - iw * SW) == 0) & (iw >= 0) & (iw < IW) & ow_mask  # [BN]

                base_valid = id_valid & ih_valid
                # combined ow validity:
                ow_valid = iw_valid & base_valid  # [BN]

                # Inner loop over ic in chunks
                for ic_start in range(0, IC, BLOCK_K):
                    ic_offs = ic_start + tl.arange(0, BLOCK_K)  # [BK]
                    ic_mask = ic_offs < IC

                    # Load x[n, ic, id_, ih, iw] -> [BK, BN]
                    # x_idx = ((n*IC + ic)*ID + id_)*IH*IW + ih*IW + iw
                    x_base = ((n * IC + ic_offs[:, None]) * ID + id_) * IH * IW + ih * IW + iw[None, :]  # [BK, BN]
                    x_load_mask = ic_mask[:, None] & ow_valid[None, :]
                    x_vals = tl.load(x_ptr + x_base, mask=x_load_mask, other=0.0)  # [BK, BN]

                    # Load weight[ic, oc, kd, kh, kw] -> [BK, BM]
                    # w_idx = ((ic*OC + oc)*KD + kd)*KH*KW + kh*KW + kw
                    w_base = ((ic_offs[:, None] * OC + oc_offs[None, :]) * KD + kd) * KH * KW + kh * KW + kw  # [BK, BM]
                    w_load_mask = ic_mask[:, None] & oc_mask[None, :]
                    w_vals = tl.load(w_ptr + w_base, mask=w_load_mask, other=0.0)  # [BK, BM]

                    # We want acc[BM, BN] += w_vals.T @ x_vals
                    # tl.dot expects [M,K] x [K,N] -> [M,N]
                    # w_vals is [BK, BM], so transpose -> [BM, BK]
                    acc += tl.dot(tl.trans(w_vals), x_vals)

    # Add bias
    bv = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BM]
    acc += bv[:, None]

    # Store: out[n, oc, od, oh, ow]
    out_base = ((n * OC + oc_offs[:, None]) * OD + od) * OH * OW + oh * OW + ow_offs[None, :]  # [BM, BN]
    out_mask = oc_mask[:, None] & ow_mask[None, :]
    tl.store(out_ptr + out_base, acc, mask=out_mask)


def conv_transpose3d_triton(x, weight, bias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    SD = SH = SW = stride
    PD = PH = PW = padding
    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        N * OD * OH,
        triton.cdiv(OC, meta['BLOCK_OC']),
        triton.cdiv(OW, meta['BLOCK_OW']),
    )

    conv_transpose3d_gemm_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
    )
    return out


@triton.jit
def bn_meansub_kernel(
    x_ptr, out_ptr,
    scale_ptr, shift_ptr,
    N, C, SPATIAL,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    scale = tl.load(scale_ptr + c)
    shift = tl.load(shift_ptr + c)

    base = (n * C + c) * SPATIAL

    sum_val = tl.zeros([], dtype=tl.float32)
    for off in range(0, SPATIAL, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < SPATIAL
        v = tl.load(x_ptr + base + idx, mask=m, other=0.0)
        nv = v * scale + shift
        sum_val += tl.sum(tl.where(m, nv, 0.0))

    mean = sum_val / SPATIAL

    for off in range(0, SPATIAL, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < SPATIAL
        v = tl.load(x_ptr + base + idx, mask=m, other=0.0)
        nv = v * scale + shift
        res = nv - mean
        tl.store(out_ptr + base + idx, res, mask=m)


def bn_meansub_triton(x, scale, shift):
    N, C, D, H, W = x.shape
    SPATIAL = D * H * W
    out = torch.empty_like(x)
    grid = (N * C,)
    BLOCK = 1024
    bn_meansub_kernel[grid](
        x, out, scale, shift,
        N, C, SPATIAL,
        BLOCK=BLOCK,
        num_warps=4,
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
        weight = self.conv_transpose.weight.contiguous()
        if self.conv_transpose.bias is not None:
            bias = self.conv_transpose.bias.contiguous()
        else:
            bias = torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)

        y = conv_transpose3d_triton(x, weight, bias, self.stride, self.padding)

        if self.training:
            dims = (0, 2, 3, 4)
            mean = y.mean(dim=dims)
            var = y.var(dim=dims, unbiased=False)
            with torch.no_grad():
                momentum = self.batch_norm.momentum
                self.batch_norm.running_mean.mul_(1 - momentum).add_(mean.detach(), alpha=momentum)
                Nn = y.shape[0] * y.shape[2] * y.shape[3] * y.shape[4]
                unbiased_var = var.detach() * Nn / (Nn - 1) if Nn > 1 else var.detach()
                self.batch_norm.running_var.mul_(1 - momentum).add_(unbiased_var, alpha=momentum)
                self.batch_norm.num_batches_tracked.add_(1)
        else:
            mean = self.batch_norm.running_mean
            var = self.batch_norm.running_var

        eps = self.batch_norm.eps
        gamma = self.batch_norm.weight
        beta = self.batch_norm.bias
        invstd = torch.rsqrt(var + eps)
        scale = (gamma * invstd).contiguous()
        shift = (beta - mean * gamma * invstd).contiguous()

        out = bn_meansub_triton(y, scale, shift)
        return out