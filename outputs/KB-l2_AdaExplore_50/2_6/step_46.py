import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=['N', 'OC', 'PD', 'PH', 'PW'],
)
@triton.jit
def fused_conv_softmax_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,  # conv output dims
    PD, PH, PW,      # pooled output dims
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    WIN: tl.constexpr,   # pool window per dim
    WIN3: tl.constexpr,  # WIN^3
    IC_C: tl.constexpr,  # in_channels (constexpr)
    OC_C: tl.constexpr,  # out_channels (constexpr)
):
    # one program per (n, pd, ph, pw)
    pid = tl.program_id(0)
    pw = pid % PW
    t1 = pid // PW
    ph = t1 % PH
    t2 = t1 // PH
    pd = t2 % PD
    n = t2 // PD

    od_start = pd * WIN
    oh_start = ph * WIN
    ow_start = pw * WIN

    c_offs = tl.arange(0, OC_C)
    c_mask = c_offs < OC
    ic_offs = tl.arange(0, IC_C)
    ic_mask = ic_offs < IC

    # bias
    bias = tl.load(b_ptr + c_offs, mask=c_mask, other=0.0)

    # Accumulator: [WIN3, OC_C] for all positions in pool window
    acc = tl.zeros((WIN3, OC_C), dtype=tl.float32)

    # Position offsets within the pool window
    pos = tl.arange(0, WIN3)
    pos_w = pos % WIN
    pos_h = (pos // WIN) % WIN
    pos_d = pos // (WIN * WIN)

    # Loop over kernel positions (weights hoisted out of position loop)
    for kd in tl.static_range(0, KD):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                # Load weight[:, :, kd, kh, kw] once: [OC_C, IC_C]
                w_base = (kd * KH + kh) * KW + kw
                w_ptrs = (w_ptr
                          + c_offs[:, None] * (IC * KD * KH * KW)
                          + ic_offs[None, :] * (KD * KH * KW)
                          + w_base)
                w_vals = tl.load(w_ptrs,
                                 mask=c_mask[:, None] & ic_mask[None, :],
                                 other=0.0)  # [OC_C, IC_C]

                # For each position in pool window, load x[n, :, od+kd, oh+kh, ow+kw]
                id_off = pos_d + kd  # [WIN3]
                ih_off = pos_h + kh
                iw_off = pos_w + kw

                # x[n, ic, od_start+id_off, oh_start+ih_off, ow_start+iw_off]
                # shape [WIN3, IC_C]
                x_base = (n * IC) * ID * IH * IW
                x_ptrs = (x_ptr + x_base
                          + ic_offs[None, :] * (ID * IH * IW)
                          + (od_start + id_off)[:, None] * (IH * IW)
                          + (oh_start + ih_off)[:, None] * IW
                          + (ow_start + iw_off)[:, None])
                x_vals = tl.load(x_ptrs, mask=ic_mask[None, :], other=0.0)  # [WIN3, IC_C]

                # acc[pos, oc] += sum_ic x_vals[pos, ic] * w_vals[oc, ic]
                # Use matmul: x_vals @ w_vals.T -> [WIN3, OC_C]
                acc += tl.dot(x_vals, tl.trans(w_vals))

    # Add bias
    acc = acc + bias[None, :]

    # Softmax over OC dim
    acc_for_sm = tl.where(c_mask[None, :], acc, -float('inf'))
    m = tl.max(acc_for_sm, axis=1)  # [WIN3]
    shifted = acc_for_sm - m[:, None]
    e = tl.exp(shifted)
    e = tl.where(c_mask[None, :], e, 0.0)
    s = tl.sum(e, axis=1)  # [WIN3]
    sm = e / s[:, None]  # [WIN3, OC_C]

    # Max-pool over WIN3 positions
    sm_for_max = tl.where(c_mask[None, :], sm, -float('inf'))
    max_vals = tl.max(sm_for_max, axis=0)  # [OC_C]

    # Store output[n, c, pd, ph, pw]
    out_base = ((n * OC) * PD + pd) * PH * PW + ph * PW + pw
    out_ptrs = out_ptr + out_base + c_offs * (PD * PH * PW)
    tl.store(out_ptrs, max_vals, mask=c_mask)


def fused_conv_softmax_pool(x, weight, bias, pool_kernel_size):
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1
    WIN = pool_kernel_size * pool_kernel_size
    PD = OD // WIN
    PH = OH // WIN
    PW = OW // WIN

    out = torch.empty((N, OC, PD, PH, PW), device=x.device, dtype=torch.float32)

    def npot(v):
        r = 1
        while r < v:
            r *= 2
        return r

    # tl.dot requires minimum sizes (typically >=16)
    IC_C = max(16, npot(IC))
    OC_C = max(16, npot(OC))
    WIN3 = WIN * WIN * WIN

    grid = (N * PD * PH * PW,)
    fused_conv_softmax_pool_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        PD, PH, PW,
        KD=KD, KH=KH, KW=KW,
        WIN=WIN,
        WIN3=WIN3,
        IC_C=IC_C,
        OC_C=OC_C,
    )
    return out


@triton.jit
def fused_softmax_pool_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    WIN: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    ow = pid % OW
    pid1 = pid // OW
    oh = pid1 % OH
    pid2 = pid1 // OH
    od = pid2 % OD
    n = pid2 // OD

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    d_start = od * WIN
    h_start = oh * WIN
    w_start = ow * WIN

    max_vals = tl.full((BLOCK_C,), -float('inf'), dtype=tl.float32)

    for dd in tl.static_range(0, WIN):
        for hh in tl.static_range(0, WIN):
            for ww in tl.static_range(0, WIN):
                d = d_start + dd
                h = h_start + hh
                w = w_start + ww
                base = ((n * C + 0) * D + d) * H * W + h * W + w
                ptrs = x_ptr + base + c_offs * (D * H * W)
                vals = tl.load(ptrs, mask=c_mask, other=-float('inf'))
                m = tl.max(vals, axis=0)
                vals_shift = vals - m
                e = tl.exp(vals_shift)
                e = tl.where(c_mask, e, 0.0)
                s = tl.sum(e, axis=0)
                sm = e / s
                sm = tl.where(c_mask, sm, -float('inf'))
                max_vals = tl.maximum(max_vals, sm)

    out_base = ((n * C + 0) * OD + od) * OH * OW + oh * OW + ow
    out_ptrs = out_ptr + out_base + c_offs * (OD * OH * OW)
    tl.store(out_ptrs, max_vals, mask=c_mask)


def fused_softmax_pool(x, pool_kernel_size):
    N, C, D, H, W = x.shape
    WIN = pool_kernel_size * pool_kernel_size
    OD = D // WIN
    OH = H // WIN
    OW = W // WIN
    out = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)
    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2
    grid = (N * OD * OH * OW,)
    fused_softmax_pool_kernel[grid](
        x, out,
        N, C, D, H, W, OD, OH, OW,
        WIN=WIN, BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        WIN = self.pool_kernel_size * self.pool_kernel_size
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1

        if (OD % WIN == 0) and (OH % WIN == 0) and (OW % WIN == 0):
            try:
                return fused_conv_softmax_pool(
                    x, self.conv.weight, self.conv.bias, self.pool_kernel_size
                )
            except Exception:
                pass

        x = self.conv(x)
        N, C, D, H, W = x.shape
        if (D % WIN == 0) and (H % WIN == 0) and (W % WIN == 0):
            return fused_softmax_pool(x.contiguous(), self.pool_kernel_size)
        x = torch.softmax(x, dim=1)
        x = F.max_pool3d(x, self.pool_kernel_size)
        x = F.max_pool3d(x, self.pool_kernel_size)
        return x