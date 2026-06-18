import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_kernel(
    x_ptr, w_ptr, conv_bias_ptr, bias_ptr, out_ptr,
    N, IC, D, H, W,
    OC, KD, KH, KW,
    OD, OH, OW,
    PD, PH, PW,  # pooled dims
    inv_div_pool,  # 1/(divisor * PD*PH*PW)
    BLOCK_OC: tl.constexpr,
    IC_KDHW: tl.constexpr,
):
    # one program per batch
    n = tl.program_id(0)

    # accumulator: sum over (oc, pd, ph, pw) of ((sum_window_max) * inv_div_pool + bias[oc])
    # Final output: sum_over_oc( global_avg(maxpool(conv/div)) + bias[oc] )
    # = sum_over_oc( bias[oc] + (1/(PD*PH*PW)) * sum_{pd,ph,pw}( max over 2x2x2 of conv[oc,...] / div ) )
    # = sum_over_oc(bias[oc]) + inv_div_pool * sum_over_oc_and_pooled_positions( max_window )

    # We'll compute total = sum_over_oc(bias) + inv_div_pool * SUM
    # where SUM = sum over (oc, pd, ph, pw) of max over 2x2x2 of conv output at (2*pd+i, 2*ph+j, 2*pw+k)

    offs_oc = tl.arange(0, BLOCK_OC)
    oc_mask = offs_oc < OC

    # Sum over all pooled spatial positions and all oc
    total_sum = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Load conv bias for all oc in tile
    cbias = tl.load(conv_bias_ptr + offs_oc, mask=oc_mask, other=0.0)

    # Iterate over pooled spatial positions
    for pd in range(0, PD):
        for ph in range(0, PH):
            for pw in range(0, PW):
                # max over 2x2x2 window in conv output
                # conv output positions: (2*pd+i, 2*ph+j, 2*pw+k) for i,j,k in {0,1}
                max_val = tl.full((BLOCK_OC,), -float('inf'), dtype=tl.float32)
                for i in tl.static_range(0, 2):
                    for j in tl.static_range(0, 2):
                        for k in tl.static_range(0, 2):
                            od = 2 * pd + i
                            oh = 2 * ph + j
                            ow = 2 * pw + k
                            # compute conv at (n, oc, od, oh, ow) for all oc in tile
                            # conv[n, oc, od, oh, ow] = sum over (ic, kd, kh, kw) of x[n, ic, od+kd, oh+kh, ow+kw] * w[oc, ic, kd, kh, kw]
                            # use precomputed weight reshaped as [OC, IC*KD*KH*KW]
                            acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)
                            # loop over IC*KD*KH*KW
                            for kk in range(0, IC_KDHW):
                                # decode kk -> (ic, kd, kh, kw)
                                ic = kk // (KD * KH * KW)
                                rem = kk % (KD * KH * KW)
                                kd = rem // (KH * KW)
                                rem2 = rem % (KH * KW)
                                kh = rem2 // KW
                                kw_ = rem2 % KW
                                # load x[n, ic, od+kd, oh+kh, ow+kw]
                                x_off = ((n * IC + ic) * D + (od + kd)) * H * W + (oh + kh) * W + (ow + kw_)
                                xv = tl.load(x_ptr + x_off)
                                # load w[oc, kk] for all oc
                                w_off = offs_oc * IC_KDHW + kk
                                wv = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                                acc += xv * wv
                            acc = acc + cbias
                            max_val = tl.maximum(max_val, acc)
                total_sum += max_val

    # total_sum is sum over pooled positions of max per oc
    # final per-oc contribution: bias[oc] + inv_div_pool * total_sum[oc]
    # then sum over oc
    bias_vals = tl.load(bias_ptr + offs_oc, mask=oc_mask, other=0.0)
    per_oc = bias_vals + inv_div_pool * total_sum
    per_oc = tl.where(oc_mask, per_oc, 0.0)
    result = tl.sum(per_oc, axis=0)
    tl.store(out_ptr + n, result)


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

        # Only handle pool_size == (2,2,2) in fused kernel; otherwise fallback
        if (PSD, PSH, PSW) != (2, 2, 2):
            return self._fallback(x)

        x = x.contiguous()
        w = self.conv.weight.contiguous().view(OC, IC * KD * KH * KW).contiguous()
        cbias = self.conv.bias.contiguous()
        bias_flat = self.bias.view(-1).contiguous()

        out = torch.empty(N, device=x.device, dtype=x.dtype)

        IC_KDHW = IC * KD * KH * KW
        BLOCK_OC = triton.next_power_of_2(OC)
        inv_div_pool = 1.0 / (self.divisor * PD * PH * PW)

        grid = (N,)
        fused_conv_kernel[grid](
            x, w, cbias, bias_flat, out,
            N, IC, D, H, W,
            OC, KD, KH, KW,
            OD, OH, OW,
            PD, PH, PW,
            inv_div_pool,
            BLOCK_OC=BLOCK_OC,
            IC_KDHW=IC_KDHW,
            num_warps=4,
        )

        # output shape: original is sum over dim=1 of [N, OC, 1, 1, 1] + bias broadcast
        # bias shape is (OC, 1, 1, 1), adding to [N, OC, 1, 1, 1] -> [N, OC, 1, 1, 1]
        # sum dim=1 -> [N, 1, 1, 1]
        return out.view(N, 1, 1, 1)

    def _fallback(self, x):
        x = self.conv(x)
        x = x / self.divisor
        x = self.max_pool(x)
        x = self.global_avg_pool(x)
        x = x + self.bias
        x = torch.sum(x, dim=self.sum_dim)
        return x