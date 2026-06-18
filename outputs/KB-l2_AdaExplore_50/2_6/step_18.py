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
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    P: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_POOL: tl.constexpr,
    BLOCK_KVOL: tl.constexpr,
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

    # Pool indices [BLOCK_POOL]
    pool_offs = tl.arange(0, BLOCK_POOL)
    pool_mask = pool_offs < (P * P * P)
    di = pool_offs // (P * P)
    hi = (pool_offs // P) % P
    wi = pool_offs % P

    # KVOL indices [BLOCK_KVOL]
    KHKW = KH * KW
    KDHW = KD * KHKW
    KVOL = IC * KDHW
    kvol_offs = tl.arange(0, BLOCK_KVOL)
    kvol_mask = kvol_offs < KVOL
    ic = kvol_offs // KDHW
    rem = kvol_offs % KDHW
    kd = rem // KHKW
    rem2 = rem % KHKW
    kh = rem2 // KW
    kw = rem2 % KW

    # Build x patch [BLOCK_POOL, BLOCK_KVOL]
    id_ = cd_base + di[:, None] + kd[None, :]
    ih_ = ch_base + hi[:, None] + kh[None, :]
    iw_ = cw_base + wi[:, None] + kw[None, :]
    x_off = (n * IC * ID * IH * IW
             + ic[None, :] * (ID * IH * IW)
             + id_ * (IH * IW)
             + ih_ * IW
             + iw_)
    x_mask = pool_mask[:, None] & kvol_mask[None, :]
    x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

    # Build weight tile [BLOCK_KVOL, BLOCK_OC]
    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC
    w_off = oc_offs[None, :] * KVOL + kvol_offs[:, None]
    w_mask = kvol_mask[:, None] & oc_mask[None, :]
    w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

    # Matmul: [BLOCK_POOL, BLOCK_OC]
    conv = tl.dot(x_tile, w_tile, allow_tf32=True)

    # Add bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    conv = conv + bias[None, :]

    # Softmax along OC per pool row
    neg_inf = float('-inf')
    conv_for_max = tl.where(oc_mask[None, :], conv, neg_inf)
    row_max = tl.max(conv_for_max, axis=1)
    ex = tl.exp(conv - row_max[:, None])
    ex = tl.where(oc_mask[None, :], ex, 0.0)
    row_sum = tl.sum(ex, axis=1)
    sm = ex / row_sum[:, None]

    # Mask invalid pool rows out before max-reduce
    sm = tl.where(pool_mask[:, None], sm, neg_inf)
    pool_max = tl.max(sm, axis=0)

    # Store output
    out_base = n * OC * OD * OH * OW + od * OH * OW + oh * OW + ow
    out_offs = out_base + oc_offs * (OD * OH * OW)
    tl.store(out_ptr + out_offs, pool_max, mask=oc_mask)


def fused_conv_softmax_pool(x, weight, bias, pool_factor):
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    CD = ID - KD + 1
    CH = IH - KH + 1
    CW = IW - KW + 1
    # Sequential pool semantics: out = (CD//2)//2 etc.
    p1 = pool_factor  # nope, pool_factor here is product
    # use sqrt assumption: pool_factor = pk*pk, sequential pools each pk
    # but to match exactly: use floor div twice
    # Here we receive total P; assume P=pk*pk and each pk=int(sqrt(P))
    OD = CD // pool_factor
    OH = CH // pool_factor
    OW = CW // pool_factor

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 16
    while BLOCK_OC < OC:
        BLOCK_OC *= 2

    P3 = pool_factor * pool_factor * pool_factor
    BLOCK_POOL = 1
    while BLOCK_POOL < P3:
        BLOCK_POOL *= 2
    BLOCK_POOL = max(BLOCK_POOL, 16)

    KVOL = IC * KD * KH * KW
    BLOCK_KVOL = 16
    while BLOCK_KVOL < KVOL:
        BLOCK_KVOL *= 2

    grid = (N * OD * OH * OW,)
    fused_conv_softmax_pool_kernel[grid](
        x, weight, bias, out,
        N, IC,
        ID, IH, IW,
        OC,
        OD, OH, OW,
        KD, KH, KW,
        pool_factor,
        BLOCK_OC,
        BLOCK_POOL,
        BLOCK_KVOL,
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
        pk = self.pool_kernel_size
        # sequential pool floor semantics
        OD = (CD // pk) // pk
        OH = (CH // pk) // pk
        OW = (CW // pk) // pk
        pf = self.pool_factor
        # Need OD*pf + KD - 1 <= ID etc. for the kernel
        ok = (OD * pf + KD - 1 <= ID) and (OH * pf + KH - 1 <= IH) and (OW * pf + KW - 1 <= IW)
        if ok and OD > 0 and OH > 0 and OW > 0:
            # call kernel with pool_factor (pf) and computed OD/OH/OW
            out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)
            BLOCK_OC = 16
            while BLOCK_OC < OC:
                BLOCK_OC *= 2
            P3 = pf * pf * pf
            BLOCK_POOL = 1
            while BLOCK_POOL < P3:
                BLOCK_POOL *= 2
            BLOCK_POOL = max(BLOCK_POOL, 16)
            KVOL = IC * KD * KH * KW
            BLOCK_KVOL = 16
            while BLOCK_KVOL < KVOL:
                BLOCK_KVOL *= 2
            grid = (N * OD * OH * OW,)
            fused_conv_softmax_pool_kernel[grid](
                x, weight, bias, out,
                N, IC,
                ID, IH, IW,
                OC,
                OD, OH, OW,
                KD, KH, KW,
                pf,
                BLOCK_OC, BLOCK_POOL, BLOCK_KVOL,
                num_warps=4, num_stages=2,
            )
            return out
        else:
            x = self.conv(x)
            x = torch.softmax(x, dim=1)
            x = F.max_pool3d(x, self.pool_kernel_size)
            x = F.max_pool3d(x, self.pool_kernel_size)
            return x