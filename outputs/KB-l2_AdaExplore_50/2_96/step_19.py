import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Gather-style: directly compute maxpool over conv_transpose output without materializing it.
# One program per (N, OC). Iterates over pooled spatial positions, for each computes max over
# the 2x2x2 maxpool window where each value is conv_transpose output at that spatial loc.
# Each value = sum over (ic, kd, kh, kw) of input[n, ic, id, ih, iw] * weight[ic, oc, kd, kh, kw]
# subject to (out_d + pad - kd) % stride == 0 and id = (out_d + pad - kd) / stride in range.
#
# For stride=2, pad=1, K=3: out_d = id*2 - 1 + kd, so for a given out_d, valid (kd, id) pairs
# satisfy id = (out_d + 1 - kd) / 2 with kd same parity as (out_d+1).

@triton.jit
def _fused_convT_maxpool_mean_kernel(
    x_ptr,        # input (N, IC, ID, IH, IW)
    w_ptr,        # weight (IC, OC, KD, KH, KW)
    b_ptr,        # bias (OC,)
    out_ptr,      # (N, OC)
    N, IC: tl.constexpr, OC: tl.constexpr,
    ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    stride: tl.constexpr, padding: tl.constexpr,
    PK: tl.constexpr,  # maxpool kernel size
    scale: tl.constexpr,
    clamp_min: tl.constexpr,
    clamp_max: tl.constexpr,
    inv_count: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid % OC

    bias_val = tl.load(b_ptr + oc)

    acc_sum = tl.zeros((1,), dtype=tl.float32)
    sum_scalar = 0.0

    # iterate over pooled positions
    for pd in tl.static_range(0, PD):
        for ph in tl.static_range(0, PH):
            for pw in tl.static_range(0, PW):
                max_val = float("-inf")
                # iterate over the maxpool window
                for kpd in tl.static_range(0, PK):
                    od = pd * PK + kpd
                    for kph in tl.static_range(0, PK):
                        oh = ph * PK + kph
                        for kpw in tl.static_range(0, PK):
                            ow = pw * PK + kpw
                            # compute convT value at (n, oc, od, oh, ow)
                            val = bias_val
                            # iterate kernel positions; only those with valid id participate
                            for kd in tl.static_range(0, KD):
                                id_num = od + padding - kd
                                id_valid = (id_num >= 0) & (id_num % stride == 0)
                                id_v = id_num // stride
                                id_in_range = (id_v >= 0) & (id_v < ID)
                                d_ok = id_valid & id_in_range
                                for kh in tl.static_range(0, KH):
                                    ih_num = oh + padding - kh
                                    ih_valid = (ih_num >= 0) & (ih_num % stride == 0)
                                    ih_v = ih_num // stride
                                    ih_in_range = (ih_v >= 0) & (ih_v < IH)
                                    h_ok = ih_valid & ih_in_range
                                    for kw in tl.static_range(0, KW):
                                        iw_num = ow + padding - kw
                                        iw_valid = (iw_num >= 0) & (iw_num % stride == 0)
                                        iw_v = iw_num // stride
                                        iw_in_range = (iw_v >= 0) & (iw_v < IW)
                                        ok = d_ok & h_ok & iw_valid & iw_in_range
                                        # sum over IC
                                        ic_offs = tl.arange(0, IC)
                                        x_off = ((n * IC + ic_offs) * ID + id_v) * IH * IW + ih_v * IW + iw_v
                                        w_off = ((ic_offs * OC + oc) * KD + kd) * KH * KW + kh * KW + kw
                                        xv = tl.load(x_ptr + x_off, mask=ok, other=0.0)
                                        wv = tl.load(w_ptr + w_off, mask=ok, other=0.0)
                                        val += tl.sum(xv * wv, axis=0)
                            scaled = val * scale
                            max_val = tl.maximum(max_val, scaled)
                sum_scalar += max_val

    mean = sum_scalar * inv_count
    mean = tl.minimum(tl.maximum(mean, clamp_min), clamp_max)
    tl.store(out_ptr + n * OC + oc, mean)


# Fallback fused maxpool+mean kernel (operates on materialized convT output)
@triton.jit
def _fused_maxpool_mean_kernel(
    x_ptr, out_ptr,
    N, C,
    D, H, W,
    PD, PH, PW,
    scale: tl.constexpr,
    clamp_min: tl.constexpr,
    clamp_max: tl.constexpr,
    inv_count,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    spatial_in = D * H * W
    base_in = (n * C + c) * spatial_in
    pooled_total = PD * PH * PW

    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)

    NEG_INF = float("-inf")

    for start in range(0, pooled_total, BLOCK):
        idx = start + offs
        mask = idx < pooled_total
        pw = idx % PW
        tmp = idx // PW
        ph = tmp % PH
        pd = tmp // PH
        d0 = pd * K
        h0 = ph * K
        w0 = pw * K

        m = tl.full((BLOCK,), NEG_INF, dtype=tl.float32)
        for kd in tl.static_range(0, K):
            for kh in tl.static_range(0, K):
                for kw in tl.static_range(0, K):
                    in_idx = (d0 + kd) * (H * W) + (h0 + kh) * W + (w0 + kw)
                    v = tl.load(x_ptr + base_in + in_idx, mask=mask, other=NEG_INF)
                    m = tl.maximum(m, v)
        m = tl.where(mask, m, 0.0)
        acc += m

    total_sum = tl.sum(acc, axis=0)
    mean = total_sum * inv_count * scale
    mean = tl.minimum(tl.maximum(mean, clamp_min), clamp_max)
    tl.store(out_ptr + n * C + c, mean)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = float(scale)
        self.maxpool_kernel_size = maxpool_kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.clamp_min = 0.0
        self.clamp_max = 1.0
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        # Use the existing fused maxpool+mean kernel after materializing convT (faster than gather here).
        x = self.conv_transpose(x)
        x = x.contiguous()
        N, C, D, H, W = x.shape
        K = self.maxpool_kernel_size
        PD = D // K
        PH = H // K
        PW = W // K
        out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)
        pooled_total = PD * PH * PW
        inv_count = 1.0 / pooled_total
        if pooled_total <= 1024:
            BLOCK = 1024
        elif pooled_total <= 2048:
            BLOCK = 2048
        else:
            BLOCK = 4096
        grid = (N * C,)
        _fused_maxpool_mean_kernel[grid](
            x, out,
            N, C,
            D, H, W,
            PD, PH, PW,
            float(self.scale),
            float(self.clamp_min),
            float(self.clamp_max),
            inv_count,
            K=K,
            BLOCK=BLOCK,
            num_warps=4,
            num_stages=2,
        )
        return out