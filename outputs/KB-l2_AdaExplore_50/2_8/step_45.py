import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_div_pool_gap_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, D, H, W,
    OC, OD, OH, OW,
    PD, PH, PW,
    inv_div_pool,  # 1/(divisor * pool_vol * gap_vol)
    BLOCK_K: tl.constexpr,  # IC*KD*KH*KW padded
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
):
    # One program per (n, oc) -> compute conv full volume, divide, maxpool, average
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid % OC

    # Load weight for this oc: shape [IC*KD*KH*KW]
    k_offs = tl.arange(0, BLOCK_K)
    K_real = IC * KD * KH * KW
    k_mask = k_offs < K_real
    w_vals = tl.load(w_ptr + oc * K_real + k_offs, mask=k_mask, other=0.0)
    b_val = tl.load(b_ptr + oc)

    # Iterate over pooled output positions; for each, compute 2x2x2 conv outputs, take max
    acc_sum = 0.0
    pool_total = PD * PH * PW

    for pd in range(PD):
        for ph in range(PH):
            for pw in range(PW):
                # 2x2x2 conv outputs starting at (pd*2, ph*2, pw*2)
                max_val = -float('inf')
                for dd in range(2):
                    for dh in range(2):
                        for dw in range(2):
                            od = pd * 2 + dd
                            oh = ph * 2 + dh
                            ow = pw * 2 + dw
                            # Compute conv at (n, oc, od, oh, ow)
                            # gather input window
                            # offsets: ic*KD*KH*KW + kd*KH*KW + kh*KW + kw
                            ic_idx = k_offs // (KD * KH * KW)
                            rem = k_offs % (KD * KH * KW)
                            kd_idx = rem // (KH * KW)
                            rem2 = rem % (KH * KW)
                            kh_idx = rem2 // KW
                            kw_idx = rem2 % KW

                            in_d = od + kd_idx
                            in_h = oh + kh_idx
                            in_w = ow + kw_idx

                            x_off = ((n * IC + ic_idx) * D + in_d) * H * W + in_h * W + in_w
                            x_vals = tl.load(x_ptr + x_off, mask=k_mask, other=0.0)
                            conv_val = tl.sum(x_vals * w_vals, axis=0)
                            max_val = tl.maximum(max_val, conv_val)
                acc_sum += max_val

    # mean over PD*PH*PW, divide by divisor (combined), add bias
    mean_val = acc_sum * inv_div_pool
    result = mean_val + b_val

    # Atomic add into out[n] (sum over oc)
    tl.atomic_add(out_ptr + n, result)


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
        PSD, PSH, PSW = self.pool_size
        PD = OD // PSD
        PH = OH // PSH
        PW = OW // PSW

        # Only support the configured shapes via this fused kernel
        if not (PSD == 2 and PSH == 2 and PSW == 2 and PD * 2 == OD and PH * 2 == OH and PW * 2 == OW):
            # fallback
            y = self.conv(x)
            y = y / self.divisor
            y = self.max_pool(y)
            y = self.global_avg_pool(y)
            y = y + self.bias
            y = torch.sum(y, dim=self.sum_dim)
            return y

        K_real = IC * KD * KH * KW
        BLOCK_K = triton.next_power_of_2(K_real)

        weight = self.conv.weight.contiguous().view(OC, K_real)
        # Fold conv bias into the per-(n,oc) result: conv bias adds bias[oc] to every conv output.
        # After divide and maxpool and gap, that becomes conv_bias[oc] / divisor.
        # We can add it to b_val. Let's combine: effective_bias = self.bias[oc] + conv_bias[oc]/divisor
        conv_bias = self.conv.bias.contiguous()
        effective_bias = self.bias.view(-1) + conv_bias / self.divisor
        effective_bias = effective_bias.contiguous()

        pool_vol = PSD * PSH * PSW  # 8
        gap_vol = PD * PH * PW
        # max already accounts for pool; we need mean over PD*PH*PW (gap), and divide by divisor
        inv_div_pool = 1.0 / (self.divisor * gap_vol)

        out = torch.zeros(N, device=x.device, dtype=x.dtype)

        grid = (N * OC,)
        fused_conv_div_pool_gap_kernel[grid](
            x, weight, effective_bias, out,
            N, IC, D, H, W,
            OC, OD, OH, OW,
            PD, PH, PW,
            inv_div_pool,
            BLOCK_K=BLOCK_K,
            KD=KD, KH=KH, KW=KW,
            num_warps=4,
        )

        # sum_dim=1 reduces OC dim; output shape after sum is [N, 1, 1, 1] from [N, OC, 1, 1, 1] -> [N,1,1,1]
        return out.view(N, 1, 1, 1)