import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_kernel(
    x_ptr, w_ptr, conv_bias_ptr, add_bias_ptr, out_ptr,
    N, IC, D, H, W,
    OC, KD, KH, KW,
    OD, OH, OW,        # conv output dims
    PD, PH, PW,        # pooled dims (OD//2, OH//2, OW//2)
    inv_divisor,
    inv_pool_count,    # 1 / (PD*PH*PW)
    BLOCK_OC: tl.constexpr,
    KVOL: tl.constexpr,
):
    # one program per batch
    n = tl.program_id(0)

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Load weight: [OC, IC*KD*KH*KW]
    # Load conv bias
    conv_b = tl.load(conv_bias_ptr + oc_offs, mask=oc_mask, other=0.0)

    # Accumulator for sum over pooled spatial positions per oc
    acc_sum = tl.zeros([BLOCK_OC], dtype=tl.float32)

    # Iterate over pooled positions
    n_pooled = PD * PH * PW
    for pidx in range(0, n_pooled):
        pd = pidx // (PH * PW)
        rem = pidx % (PH * PW)
        ph = rem // PW
        pw = rem % PW

        # 2x2x2 max over conv outputs
        max_val = tl.full([BLOCK_OC], -float('inf'), dtype=tl.float32)

        for sub in range(0, 8):
            sd = sub // 4
            sr = sub % 4
            sh = sr // 2
            sw = sr % 2
            od = pd * 2 + sd
            oh = ph * 2 + sh
            ow = pw * 2 + sw

            # Compute conv output for (n, oc, od, oh, ow) via GEMM-K loop
            conv_val = tl.zeros([BLOCK_OC], dtype=tl.float32)

            for k in range(0, KVOL):
                # k = ic * KD*KH*KW + kd*KH*KW + kh*KW + kw
                ic = k // (KD * KH * KW)
                krem = k % (KD * KH * KW)
                kd = krem // (KH * KW)
                krem2 = krem % (KH * KW)
                kh = krem2 // KW
                kw = krem2 % KW

                id_ = od + kd
                ih = oh + kh
                iw = ow + kw

                x_off = ((n * IC + ic) * D + id_) * H * W + ih * W + iw
                x_val = tl.load(x_ptr + x_off)

                # Weight: [OC, IC*KD*KH*KW]
                w_off = oc_offs * (IC * KD * KH * KW) + k
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)

                conv_val += w_val * x_val

            conv_val = (conv_val + conv_b) * inv_divisor
            max_val = tl.maximum(max_val, conv_val)

        acc_sum += max_val

    # Global avg pool -> divide by n_pooled
    avg = acc_sum * inv_pool_count
    # add bias
    bias_val = tl.load(add_bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    final = avg + bias_val
    # sum over oc
    final = tl.where(oc_mask, final, 0.0)
    result = tl.sum(final, axis=0)

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
        x = x.contiguous().cuda()
        N, IC, D, H, W = x.shape
        KD, KH, KW = self.kernel_size
        OC = self.out_channels
        OD = D - KD + 1
        OH = H - KH + 1
        OW = W - KW + 1
        PD = OD // self.pool_size[0]
        PH = OH // self.pool_size[1]
        PW = OW // self.pool_size[2]

        # weight reshape [OC, IC*KD*KH*KW]
        w = self.conv.weight.contiguous().view(OC, IC * KD * KH * KW).contiguous()
        conv_b = self.conv.bias.contiguous()
        add_b = self.bias.view(-1).contiguous()

        out = torch.empty(N, device=x.device, dtype=x.dtype)

        BLOCK_OC = triton.next_power_of_2(OC)
        if BLOCK_OC < 16:
            BLOCK_OC = 16
        KVOL = IC * KD * KH * KW

        inv_div = 1.0 / float(self.divisor)
        inv_pool_count = 1.0 / float(PD * PH * PW)

        grid = (N,)
        fused_conv_kernel[grid](
            x, w, conv_b, add_b, out,
            N, IC, D, H, W,
            OC, KD, KH, KW,
            OD, OH, OW,
            PD, PH, PW,
            inv_div, inv_pool_count,
            BLOCK_OC=BLOCK_OC,
            KVOL=KVOL,
            num_warps=4,
        )

        # sum_dim = 1 -> originally output shape after sum over channels: [N, 1, 1, 1]
        return out.view(N, 1, 1, 1)