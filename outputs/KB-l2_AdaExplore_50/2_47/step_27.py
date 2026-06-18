import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 512, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['N', 'OC', 'IC', 'KD', 'KH', 'KW', 'OD', 'OH', 'OW'],
)
@triton.jit
def conv3d_mish_tanh_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    # strides for x (NCDHW)
    x_sn, x_sc, x_sd, x_sh, x_sw,
    # strides for out (NCDHW)
    o_sn, o_sc, o_sd, o_sh, o_sw,
    BLOCK_M: tl.constexpr,  # OC tile
    BLOCK_N: tl.constexpr,  # spatial tile
    BLOCK_K: tl.constexpr,  # IC tile
):
    pid_m = tl.program_id(0)  # OC tile
    pid_n = tl.program_id(1)  # spatial tile
    pid_b = tl.program_id(2)  # batch

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # OC indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # spatial indices in OD*OH*OW

    OHW = OH * OW
    ODHW = OD * OHW

    # decompose spatial -> (od, oh, ow)
    od = offs_n // OHW
    rem = offs_n % OHW
    oh = rem // OW
    ow = rem % OW

    mask_m = offs_m < OC
    mask_n = offs_n < ODHW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # weight layout: (OC, IC, KD, KH, KW), contiguous
    # weight stride: w[oc, ic, kd, kh, kw] = oc*IC*KD*KH*KW + ic*KD*KH*KW + kd*KH*KW + kh*KW + kw
    IC_KDHW = IC * KD * KH * KW
    KDHW = KD * KH * KW
    KHW = KH * KW

    # x base for this batch
    x_batch = x_ptr + pid_b * x_sn

    for kd in tl.static_range(0, KD):
        id_ = od + kd  # padding=0
        for kh in tl.static_range(0, KH):
            ih = oh + kh
            for kw in tl.static_range(0, KW):
                iw = ow + kw
                # input position
                # iterate over IC in BLOCK_K tiles
                k_kdhw = kd * KHW + kh * KW + kw
                for ic0 in range(0, IC, BLOCK_K):
                    offs_k = ic0 + tl.arange(0, BLOCK_K)
                    mask_k = offs_k < IC

                    # load x[b, offs_k, id_, ih, iw] -> shape (BLOCK_K, BLOCK_N)
                    x_offs = (offs_k[:, None] * x_sc
                              + id_[None, :] * x_sd
                              + ih[None, :] * x_sh
                              + iw[None, :] * x_sw)
                    x_mask = mask_k[:, None] & mask_n[None, :]
                    x_vals = tl.load(x_batch + x_offs, mask=x_mask, other=0.0)

                    # load w[offs_m, offs_k, kd, kh, kw] -> shape (BLOCK_M, BLOCK_K)
                    w_offs = (offs_m[:, None] * IC_KDHW
                              + offs_k[None, :] * KDHW
                              + k_kdhw)
                    w_mask = mask_m[:, None] & mask_k[None, :]
                    w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

                    acc += tl.dot(w_vals, x_vals)

    # bias
    b_vals = tl.load(b_ptr + offs_m, mask=mask_m, other=0.0)
    acc = acc + b_vals[:, None]

    # Mish: a * tanh(softplus(a)); softplus = log(1+exp(a))
    # Use numerically stable softplus
    sp = tl.where(acc > 20.0, acc, tl.log(1.0 + tl.exp(acc)))
    e2 = tl.exp(2.0 * sp)
    th_sp = (e2 - 1.0) / (e2 + 1.0)
    mish = acc * th_sp
    e2m = tl.exp(2.0 * mish)
    out = (e2m - 1.0) / (e2m + 1.0)

    # store
    out_batch = out_ptr + pid_b * o_sn
    out_offs = (offs_m[:, None] * o_sc
                + od[None, :] * o_sd
                + oh[None, :] * o_sh
                + ow[None, :] * o_sw)
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_batch + out_offs, out, mask=out_mask)


def conv3d_mish_tanh(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda and b.is_cuda
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()
    N, IC, ID, IH, IW = x.shape
    OC, IC_w, KD, KH, KW = w.shape
    assert IC == IC_w
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    ODHW = OD * OH * OW

    grid = lambda meta: (
        triton.cdiv(OC, meta['BLOCK_M']),
        triton.cdiv(ODHW, meta['BLOCK_N']),
        N,
    )

    conv3d_mish_tanh_kernel[grid](
        x, w, b, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        assert stride == 1 and padding == 0, "this kernel only handles stride=1, padding=0"
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return conv3d_mish_tanh(x, self.conv.weight, self.conv.bias)