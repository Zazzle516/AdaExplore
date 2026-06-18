import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_softmax_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr,
    ID, IH, IW,
    OC: tl.constexpr,
    CD, CH, CW,
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    P: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_K: tl.constexpr,
    WIN: tl.constexpr,  # P*P*P
):
    pid = tl.program_id(0)
    ow = pid % OW
    tmp = pid // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    n = tmp // OD

    cd_base = od * P
    ch_base = oh * P
    cw_base = ow * P

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC
    k_offs = tl.arange(0, BLOCK_K)
    KVOL = IC * KD * KH * KW
    k_mask = k_offs < KVOL

    # Load weight tile [KVOL_padded, OC]: weight layout (OC, IC, KD, KH, KW)
    # w[oc, k] at oc*KVOL + k -> transposed: w_t[k, oc]
    w_ptrs = w_ptr + oc_offs[None, :] * KVOL + k_offs[:, None]
    w_mask = k_mask[:, None] & oc_mask[None, :]
    w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_OC]

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]

    # Build patch matrix [WIN, BLOCK_K]
    # For each pool position p in [0,WIN), and k in [0,KVOL):
    #   decompose p -> (di, hi, wi); decompose k -> (ic, kd, kh, kw)
    #   x offset = n*IC*ID*IH*IW + ic*ID*IH*IW + (cd_base+di+kd)*IH*IW + (ch_base+hi+kh)*IW + (cw_base+wi+kw)
    p_offs = tl.arange(0, WIN)
    wi = p_offs % P
    tmp_p = p_offs // P
    hi = tmp_p % P
    di = tmp_p // P

    kw_k = k_offs % KW
    tmp_k = k_offs // KW
    kh_k = tmp_k % KH
    tmp_k = tmp_k // KH
    kd_k = tmp_k % KD
    ic_k = tmp_k // KD

    x_n_base = n * IC * ID * IH * IW

    # d index = (cd_base + di) + kd
    d_idx = (cd_base + di)[:, None] + kd_k[None, :]  # [WIN, BLOCK_K]
    h_idx = (ch_base + hi)[:, None] + kh_k[None, :]
    w_idx = (cw_base + wi)[:, None] + kw_k[None, :]
    c_idx = ic_k[None, :]  # [1, BLOCK_K]

    x_offs = (x_n_base
              + c_idx * (ID * IH * IW)
              + d_idx * (IH * IW)
              + h_idx * IW
              + w_idx)  # [WIN, BLOCK_K]
    x_mask = k_mask[None, :]
    patch = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)  # [WIN, BLOCK_K]

    # GEMM: [WIN, BLOCK_K] x [BLOCK_K, BLOCK_OC] -> [WIN, BLOCK_OC]
    conv_out = tl.dot(patch, w_tile)
    conv_out = conv_out + bias[None, :]

    # Mask invalid OC
    conv_out = tl.where(oc_mask[None, :], conv_out, -float('inf'))

    # Softmax along OC (axis=1)
    m = tl.max(conv_out, axis=1)  # [WIN]
    ex = tl.exp(conv_out - m[:, None])
    ex = tl.where(oc_mask[None, :], ex, 0.0)
    s = tl.sum(ex, axis=1)  # [WIN]
    sm = ex / s[:, None]  # [WIN, BLOCK_OC]

    # Max-pool over WIN axis
    max_vals = tl.max(sm, axis=0)  # [BLOCK_OC]

    out_base = n * OC * OD * OH * OW + od * OH * OW + oh * OW + ow
    out_offs = out_base + oc_offs * (OD * OH * OW)
    tl.store(out_ptr + out_offs, max_vals, mask=oc_mask)


def fused_conv_softmax_pool(x, weight, bias, pool_factor):
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    CD = ID - KD + 1
    CH = IH - KH + 1
    CW = IW - KW + 1
    OD = CD // pool_factor
    OH = CH // pool_factor
    OW = CW // pool_factor

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 1
    while BLOCK_OC < OC:
        BLOCK_OC *= 2
    BLOCK_OC = max(BLOCK_OC, 16)

    KVOL = IC * KD * KH * KW
    BLOCK_K = 1
    while BLOCK_K < KVOL:
        BLOCK_K *= 2
    BLOCK_K = max(BLOCK_K, 16)

    WIN = pool_factor * pool_factor * pool_factor

    grid = (N * OD * OH * OW,)
    fused_conv_softmax_pool_kernel[grid](
        x, weight, bias, out,
        N, IC,
        ID, IH, IW,
        OC,
        CD, CH, CW,
        OD, OH, OW,
        KD, KH, KW,
        pool_factor,
        BLOCK_OC,
        BLOCK_K,
        WIN,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self.pool_factor = pool_kernel_size * pool_kernel_size

    def forward(self, x):
        x = x.contiguous()
        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()
        N, IC, ID, IH, IW = x.shape
        OC, _, KD, KH, KW = weight.shape
        CD = ID - KD + 1
        CH = IH - KH + 1
        CW = IW - KW + 1
        pf = self.pool_factor
        if CD % pf == 0 and CH % pf == 0 and CW % pf == 0:
            return fused_conv_softmax_pool(x, weight, bias, pf)
        else:
            x = self.conv(x)
            x = torch.softmax(x, dim=1)
            x = F.max_pool3d(x, self.pool_kernel_size)
            x = F.max_pool3d(x, self.pool_kernel_size)
            return x