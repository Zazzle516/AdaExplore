import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _convt3d_fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, OC: tl.constexpr,
    ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,  # conv-T out dims
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,  # pooled dims
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PADDING: tl.constexpr, POOLK: tl.constexpr,
    scale: tl.constexpr,
    clamp_min: tl.constexpr,
    clamp_max: tl.constexpr,
    inv_count: tl.constexpr,
):
    # one program per (n, oc)
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid % OC

    acc_sum = tl.zeros((1,), dtype=tl.float32)

    # Loop over pooled spatial locations
    # For each pooled (pd,ph,pw), compute max over POOLK^3 conv-T outputs
    for pd in tl.static_range(0, PD):
        for ph in tl.static_range(0, PH):
            for pw in tl.static_range(0, PW):
                m_val = tl.full((1,), -1e30, dtype=tl.float32)
                # iterate over the pool window
                for kpd in tl.static_range(0, POOLK):
                    od = pd * POOLK + kpd
                    for kph in tl.static_range(0, POOLK):
                        oh = ph * POOLK + kph
                        for kpw in tl.static_range(0, POOLK):
                            ow = pw * POOLK + kpw
                            # compute conv-T output at (n, oc, od, oh, ow)
                            # convT3d: out[od,oh,ow] = sum_{ic,kd,kh,kw}
                            #   x[ic, id, ih, iw] * w[ic, oc, kd, kh, kw]
                            # where id*stride - padding + kd = od => id = (od + padding - kd) / stride
                            v = tl.zeros((1,), dtype=tl.float32)
                            if b_ptr is not None:
                                v += tl.load(b_ptr + oc)
                            for kd in tl.static_range(0, KD):
                                id_num = od + PADDING - kd
                                id_ = id_num // STRIDE
                                id_valid = (id_num % STRIDE == 0) & (id_ >= 0) & (id_ < ID)
                                for kh in tl.static_range(0, KH):
                                    ih_num = oh + PADDING - kh
                                    ih_ = ih_num // STRIDE
                                    ih_valid = (ih_num % STRIDE == 0) & (ih_ >= 0) & (ih_ < IH)
                                    for kw in tl.static_range(0, KW):
                                        iw_num = ow + PADDING - kw
                                        iw_ = iw_num // STRIDE
                                        iw_valid = (iw_num % STRIDE == 0) & (iw_ >= 0) & (iw_ < IW)
                                        valid = id_valid & ih_valid & iw_valid
                                        # sum over IC
                                        ic_offs = tl.arange(0, IC)
                                        x_off = ((n * IC + ic_offs) * ID + id_) * IH * IW + ih_ * IW + iw_
                                        w_off = ((ic_offs * OC + oc) * KD + kd) * KH * KW + kh * KW + kw
                                        x_v = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                                        w_v = tl.load(w_ptr + w_off, mask=valid, other=0.0)
                                        v += tl.sum(x_v * w_v, axis=0)
                            v = v * scale
                            m_val = tl.maximum(m_val, v)
                acc_sum += m_val

    mean = acc_sum * inv_count
    mean = tl.minimum(tl.maximum(mean, clamp_min), clamp_max)
    tl.store(out_ptr + n * OC + oc, tl.sum(mean, axis=0))


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
        self.clamp_min = 0.0
        self.clamp_max = 1.0
        self.out_channels = out_channels

    def forward(self, x):
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