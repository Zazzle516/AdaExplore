import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OD', 'OH', 'OW', 'IC', 'KD', 'KH', 'KW'],
)
@triton.jit
def conv3d_mish_tanh_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wd, stride_wh, stride_ww,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # OC tile
    pid_n = tl.program_id(1)  # spatial tile (across OD*OH*OW)
    pid_b = tl.program_id(2)  # batch index

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # output channels
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # spatial linear idx

    mask_m = offs_m < OC
    mask_n = offs_n < (OD * OH * OW)

    # Decompose spatial linear index -> (od, oh, ow)
    ow = offs_n % OW
    tmp = offs_n // OW
    oh = tmp % OH
    od = tmp // OH

    K = IC * KD * KH * KW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # decompose k -> ic, kd, kh, kw
        kw_k = offs_k % KW
        t1 = offs_k // KW
        kh_k = t1 % KH
        t2 = t1 // KH
        kd_k = t2 % KD
        ic_k = t2 // KD

        # Weight: [OC, IC, KD, KH, KW] -> [BLOCK_M, BLOCK_K]
        w_offs = (offs_m[:, None] * stride_wo +
                  ic_k[None, :] * stride_wi +
                  kd_k[None, :] * stride_wd +
                  kh_k[None, :] * stride_wh +
                  kw_k[None, :] * stride_ww)
        w_mask = mask_m[:, None] & mask_k[None, :]
        w = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

        # Input coords: id = od + kd, ih = oh + kh, iw = ow + kw (no padding/stride)
        id_ = od[:, None] + kd_k[None, :]   # wait: need [BLOCK_N, BLOCK_K]
        # offs_n is along N dim => spatial is N axis. We want input [BLOCK_K, BLOCK_N]
        # Recompute with proper broadcasting:
        id2 = od[None, :] + kd_k[:, None]  # [BLOCK_K, BLOCK_N]
        ih2 = oh[None, :] + kh_k[:, None]
        iw2 = ow[None, :] + kw_k[:, None]
        ic2 = ic_k[:, None] + tl.zeros([BLOCK_K, BLOCK_N], dtype=tl.int32)

        x_offs = (pid_b * stride_xn +
                  ic2 * stride_xc +
                  id2 * stride_xd +
                  ih2 * stride_xh +
                  iw2 * stride_xw)
        x_mask = mask_k[:, None] & mask_n[None, :]
        x = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

        acc += tl.dot(w, x)

    # Bias
    bias = tl.load(b_ptr + offs_m, mask=mask_m, other=0.0)
    acc += bias[:, None]

    # mish(x) = x * tanh(softplus(x))
    sp = tl.log(1.0 + tl.exp(acc))
    t1v = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    m = acc * t1v
    t2v = 2.0 * tl.sigmoid(2.0 * m) - 1.0

    # Store
    out_offs = (pid_b * stride_on +
                offs_m[:, None] * stride_oc +
                od[None, :] * stride_od +
                oh[None, :] * stride_oh +
                ow[None, :] * stride_ow)
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offs, t2v, mask=out_mask)


def conv3d_mish_tanh(x, weight, bias):
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1
    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(OC, meta['BLOCK_M']),
        triton.cdiv(OD * OH * OW, meta['BLOCK_N']),
        N,
    )

    conv3d_mish_tanh_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        assert stride == 1 and padding == 0, "Custom kernel supports stride=1, padding=0"
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return conv3d_mish_tanh(x, self.conv.weight, self.conv.bias)