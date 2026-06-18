import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_softmax_maxpool_kernel(
    x_ptr,          # input: (N, IC, ID, IH, IW)
    w_ptr,          # weight: (OC, IC, KD, KH, KW)
    b_ptr,          # bias: (OC,)
    out_ptr,        # output: (N, OC, Do, Ho, Wo)
    N, IC, ID, IH, IW,
    OD, OH, OW,     # conv output dims (ID-2, IH-2, IW-2 for k=3)
    Do, Ho, Wo,     # final dims (OD/4, OH/4, OW/4)
    OC: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
):
    # Each program: one (n, do, ho, wo) output element.
    # Computes conv output for 4x4x4 spatial window x OC channels,
    # applies channel-softmax at each spatial position, takes max over window per channel,
    # then... wait: model applies softmax->pool->pool. Pool is per-channel max.
    # So result[n,c,do,ho,wo] = max over 4x4x4 window of softmax(conv_out[n,:,d,h,w])[c]
    # But we need ALL OC outputs at this spatial location.
    pid = tl.program_id(0)
    n = tl.program_id(1)

    total = Do * Ho * Wo
    if pid >= total:
        return

    wo = pid % Wo
    ho = (pid // Wo) % Ho
    do = pid // (Wo * Ho)

    d0 = do * 4
    h0 = ho * 4
    w0 = wo * 4

    oc_offs = tl.arange(0, OC)

    # Load bias for all OC
    bias = tl.load(b_ptr + oc_offs)  # (OC,)

    # Per-channel running max over 4x4x4 window of softmax outputs
    max_vals = tl.full([OC], -float('inf'), dtype=tl.float32)

    # Iterate over 4x4x4 = 64 spatial positions in conv output
    for k in tl.static_range(0, 64):
        dk = k // 16
        hk = (k // 4) % 4
        wk = k % 4
        od = d0 + dk
        oh = h0 + hk
        ow = w0 + wk

        # Compute conv output at (n, :, od, oh, ow) -> (OC,) vector
        acc = bias

        # Loop over kernel positions; IC is small (3) - flatten
        for kd in tl.static_range(0, KD):
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    id_ = od + kd
                    ih_ = oh + kh
                    iw_ = ow + kw
                    # Loop over input channels
                    for ic in tl.static_range(0, IC_C):
                        # x[n, ic, id_, ih_, iw_]
                        x_idx = ((n * IC + ic) * ID + id_) * IH * IW + ih_ * IW + iw_
                        x_val = tl.load(x_ptr + x_idx)
                        # w[:, ic, kd, kh, kw] for all OC
                        w_idx = ((oc_offs * IC + ic) * KD + kd) * KH * KW + kh * KW + kw
                        w_val = tl.load(w_ptr + w_idx)
                        acc = acc + x_val * w_val

        # Softmax across OC
        m = tl.max(acc, axis=0)
        e = tl.exp(acc - m)
        s = tl.sum(e, axis=0)
        sm = e / s

        max_vals = tl.maximum(max_vals, sm)

    # Store result: out[n, :, do, ho, wo]
    out_base = ((n * OC + 0) * Do + do) * Ho * Wo + ho * Wo + wo
    out_ptrs = out_ptr + out_base + oc_offs * (Do * Ho * Wo)
    tl.store(out_ptrs, max_vals)


def fused_conv_softmax_pool(x, weight, bias, pool_total=4):
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape

    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1

    Do = OD // pool_total
    Ho = OH // pool_total
    Wo = OW // pool_total

    out = torch.empty((N, OC, Do, Ho, Wo), device=x.device, dtype=torch.float32)

    total = Do * Ho * Wo
    grid = (total, N)

    fused_conv_softmax_maxpool_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OD, OH, OW,
        Do, Ho, Wo,
        OC=OC,
        KD=KD, KH=KH, KW=KW,
        IC_C=IC,
        num_warps=2,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self.pool_total = pool_kernel_size * pool_kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()
        return fused_conv_softmax_pool(x, weight, bias, pool_total=self.pool_total)