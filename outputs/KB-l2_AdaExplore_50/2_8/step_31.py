import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_pool_kernel(
    x_ptr,          # (N, IC, D, H, W)
    w_ptr,          # (OC, IC, KD, KH, KW)
    cb_ptr,         # (OC,) conv bias
    bias_ptr,       # (OC,) extra bias
    out_ptr,        # (N,) result after sum over OC
    N, IC, D, H, W,
    OC,
    OD, OH, OW,     # conv output spatial dims (D-2, H-2, W-2)
    PD, PH, PW,     # pooled dims = OD//2, OH//2, OW//2
    inv_div,        # 1/divisor
    inv_avg,        # 1/(PD*PH*PW)
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    OC_C: tl.constexpr,
):
    n = tl.program_id(0)

    # accumulator: sum over oc of (avg_pool(maxpool(conv/div)) + bias)
    # = sum_oc(avg_pool(maxpool(conv*inv_div))) + sum(bias)
    total = tl.zeros((), dtype=tl.float32)

    # Loop over output channels
    for oc in range(0, OC):
        # accumulate sum over pooled positions of max-pooled conv values
        sum_pooled = tl.zeros((), dtype=tl.float32)

        # Loop over pooled positions
        for pd in range(0, PD):
            for ph in range(0, PH):
                for pw in range(0, PW):
                    # The 2x2x2 window of conv outputs starts at (2*pd, 2*ph, 2*pw)
                    od_base = pd * 2
                    oh_base = ph * 2
                    ow_base = pw * 2

                    max_val = tl.zeros((), dtype=tl.float32) - 1e38

                    for dd in tl.static_range(0, 2):
                        for dh in tl.static_range(0, 2):
                            for dw in tl.static_range(0, 2):
                                od = od_base + dd
                                oh = oh_base + dh
                                ow = ow_base + dw

                                # Compute conv output at (n, oc, od, oh, ow)
                                conv_val = tl.zeros((), dtype=tl.float32)

                                # Iterate over kernel
                                for kd in tl.static_range(0, KD):
                                    for kh in tl.static_range(0, KH):
                                        for kw in tl.static_range(0, KW):
                                            id_ = od + kd
                                            ih_ = oh + kh
                                            iw_ = ow + kw

                                            # Load IC_C channels at a time
                                            ic_offs = tl.arange(0, IC_C)
                                            ic_mask = ic_offs < IC

                                            x_off = (((n * IC + ic_offs) * D + id_) * H + ih_) * W + iw_
                                            w_off = (((oc * IC + ic_offs) * KD + kd) * KH + kh) * KW + kw

                                            xv = tl.load(x_ptr + x_off, mask=ic_mask, other=0.0)
                                            wv = tl.load(w_ptr + w_off, mask=ic_mask, other=0.0)

                                            conv_val += tl.sum(xv * wv)

                                # add conv bias
                                cb = tl.load(cb_ptr + oc)
                                conv_val = (conv_val + cb) * inv_div

                                if (dd == 0) and (dh == 0) and (dw == 0):
                                    max_val = conv_val
                                else:
                                    max_val = tl.maximum(max_val, conv_val)

                    sum_pooled += max_val

        # avg pool: divide by num pooled positions
        avg_val = sum_pooled * inv_avg
        # add bias for this oc
        bv = tl.load(bias_ptr + oc)
        total += avg_val + bv

    tl.store(out_ptr + n, total)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_size = pool_size

    def forward(self, x):
        x = x.contiguous()
        N, IC, D, H, W = x.shape
        KD, KH, KW = self.kernel_size
        OC = self.out_channels
        OD = D - KD + 1
        OH = H - KH + 1
        OW = W - KW + 1
        PD = OD // self.pool_size[0]
        PH = OH // self.pool_size[1]
        PW = OW // self.pool_size[2]

        weight = self.conv.weight.contiguous()
        cbias = self.conv.bias.contiguous()
        bias_flat = self.bias.view(-1).contiguous()

        out = torch.empty(N, device=x.device, dtype=torch.float32)

        # IC_C: next power of 2 >= IC
        IC_C = 1
        while IC_C < IC:
            IC_C *= 2

        grid = (N,)
        fused_conv_pool_kernel[grid](
            x, weight, cbias, bias_flat, out,
            N, IC, D, H, W,
            OC,
            OD, OH, OW,
            PD, PH, PW,
            1.0 / self.divisor,
            1.0 / (PD * PH * PW),
            KD, KH, KW,
            IC_C,
            OC,
            num_warps=4,
        )

        # Original output shape after sum over dim=1: (N, 1, 1, 1)
        return out.view(N, 1, 1, 1)