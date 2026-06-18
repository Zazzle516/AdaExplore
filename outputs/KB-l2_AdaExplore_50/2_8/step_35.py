import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_kernel(
    x_ptr, w_ptr, b_ptr, bias_ptr, out_ptr,
    N, IC: tl.constexpr,
    ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    OC: tl.constexpr,
    inv_div,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
):
    # One program per batch element n.
    n = tl.program_id(0)

    # Load entire weight tensor [OC, IC, KD, KH, KW] = [16, 8, 3, 3, 3] = 3456 floats
    # Layout it as [OC, IC*KD*KH*KW]
    K = IC * KD * KH * KW  # 8*27 = 216
    oc_range = tl.arange(0, OC)  # [OC]
    k_range = tl.arange(0, K)    # [K]
    w_offs = oc_range[:, None] * K + k_range[None, :]
    w_tile = tl.load(w_ptr + w_offs)  # [OC, K]

    b_vals = tl.load(b_ptr + oc_range)  # [OC]
    bias_vals = tl.load(bias_ptr + oc_range)  # [OC]

    # Accumulator [OC] for sum over pooled positions (we'll divide later)
    acc = tl.zeros([OC], dtype=tl.float32)

    pool_count = PD * PH * PW

    # Iterate over pooled positions
    for pd in tl.static_range(0, PD):
        for ph in tl.static_range(0, PH):
            for pw in tl.static_range(0, PW):
                # Maxpool window 2x2x2 over conv outputs starting at (2*pd, 2*ph, 2*pw)
                max_val = tl.full([OC], -1e38, dtype=tl.float32)
                for dd in tl.static_range(0, 2):
                    for dh in tl.static_range(0, 2):
                        for dw in tl.static_range(0, 2):
                            od = pd * 2 + dd
                            oh = ph * 2 + dh
                            ow = pw * 2 + dw
                            # Build the input patch [K] for this conv output position
                            # K iterates: ic, kd, kh, kw
                            # Compute offset into x
                            ic_idx = k_range // (KD * KH * KW)
                            rem1 = k_range % (KD * KH * KW)
                            kd_idx = rem1 // (KH * KW)
                            rem2 = rem1 % (KH * KW)
                            kh_idx = rem2 // KW
                            kw_idx = rem2 % KW
                            id_ = od + kd_idx
                            ih_ = oh + kh_idx
                            iw_ = ow + kw_idx
                            x_off = ((n * IC + ic_idx) * ID + id_) * IH * IW + ih_ * IW + iw_
                            x_patch = tl.load(x_ptr + x_off)  # [K]
                            # conv: [OC, K] @ [K] -> [OC]
                            conv_val = tl.sum(w_tile * x_patch[None, :], axis=1)
                            conv_val = (conv_val + b_vals) * inv_div
                            max_val = tl.maximum(max_val, conv_val)
                acc += max_val

    avg = acc / pool_count + bias_vals  # [OC]
    # sum along channel dim
    total = tl.sum(avg, axis=0)
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
        N, IC, ID, IH, IW = x.shape
        KD, KH, KW = self.kernel_size
        OC = self.out_channels
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1
        PD = OD // 2
        PH = OH // 2
        PW = OW // 2

        x = x.contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        bias = self.bias.view(-1).contiguous()

        out = torch.zeros(N, device=x.device, dtype=x.dtype)

        grid = (N,)
        fused_kernel[grid](
            x, w, b, bias, out,
            N, IC, ID, IH, IW,
            OD, OH, OW,
            PD, PH, PW,
            OC,
            1.0 / self.divisor,
            KD, KH, KW,
            num_warps=4,
        )
        # Output shape: after sum over dim=1 of [N, OC, 1, 1, 1] -> [N, 1, 1, 1]
        return out.view(N, 1, 1, 1)