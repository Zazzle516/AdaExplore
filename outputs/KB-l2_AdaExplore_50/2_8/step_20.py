import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_kernel(
    x_ptr, w_ptr, cb_ptr, bias_ptr, out_ptr,
    N, IC, D, H, W,
    OC, KD, KH, KW,
    OD, OH, OW,           # conv output dims
    PD, PH, PW,           # pooled dims (after maxpool 2x2x2)
    divisor,
    NSPATIAL,             # PD*PH*PW
    BLOCK_OC: tl.constexpr,
    K: tl.constexpr,      # IC*KD*KH*KW
):
    n = tl.program_id(0)

    offs_oc = tl.arange(0, BLOCK_OC)
    offs_k = tl.arange(0, K)

    # Load weight tile [OC, K]  (assumes BLOCK_OC == OC)
    oc_mask = offs_oc < OC
    w = tl.load(w_ptr + offs_oc[:, None] * K + offs_k[None, :],
                mask=oc_mask[:, None], other=0.0)

    # Load conv bias [OC]
    cb = tl.load(cb_ptr + offs_oc, mask=oc_mask, other=0.0)

    # accumulator: per-oc sum across pool tiles
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Precompute k-indexing
    # k = ((ic*KD + kd)*KH + kh)*KW + kw
    kw_idx = offs_k % KW
    tmp = offs_k // KW
    kh_idx = tmp % KH
    tmp2 = tmp // KH
    kd_idx = tmp2 % KD
    ic_idx = tmp2 // KD

    # Iterate over pooled output (PD, PH, PW); pool size 2x2x2 -> 8 conv points
    for pd in range(0, PD):
        for ph in range(0, PH):
            for pw in range(0, PW):
                # Compute 8 conv outputs, take max, divide by divisor
                max_vals = tl.full((BLOCK_OC,), -float('inf'), dtype=tl.float32)
                for dd in range(0, 2):
                    for hh in range(0, 2):
                        for ww in range(0, 2):
                            od = pd * 2 + dd
                            oh = ph * 2 + hh
                            ow = pw * 2 + ww
                            # Gather input patch [K]
                            d_in = od + kd_idx
                            h_in = oh + kh_idx
                            w_in = ow + kw_idx
                            x_off = (((n * IC + ic_idx) * D + d_in) * H + h_in) * W + w_in
                            x_vals = tl.load(x_ptr + x_off).to(tl.float32)
                            # Conv: out[oc] = sum_k w[oc,k] * x[k] + cb[oc]
                            # Use dot via broadcast
                            prod = w * x_vals[None, :]
                            conv_out = tl.sum(prod, axis=1) + cb
                            max_vals = tl.maximum(max_vals, conv_out)
                # Divide by divisor (after maxpool — equivalent since /c then max == max then /c for c>0)
                pooled = max_vals / divisor
                acc += pooled

    # global avg pool: divide by NSPATIAL
    avg = acc / NSPATIAL
    # add bias
    bias_vals = tl.load(bias_ptr + offs_oc, mask=oc_mask, other=0.0)
    avg = avg + bias_vals
    # sum over oc
    avg = tl.where(oc_mask, avg, 0.0)
    result = tl.sum(avg, axis=0)

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
        K = IC * KD * KH * KW
        NSPATIAL = PD * PH * PW

        # Fallback if dims not aligned with pool
        if PD * 2 != OD or PH * 2 != OH or PW * 2 != OW:
            y = self.conv(x)
            y = y / self.divisor
            y = self.max_pool(y)
            y = self.global_avg_pool(y)
            y = y + self.bias
            y = torch.sum(y, dim=self.sum_dim)
            return y

        weight = self.conv.weight.contiguous().view(OC, K).contiguous()
        cbias = self.conv.bias.contiguous() if self.conv.bias is not None else torch.zeros(OC, device=x.device, dtype=x.dtype)
        bias_flat = self.bias.view(-1).contiguous()

        out = torch.empty(N, device=x.device, dtype=torch.float32)

        # Choose BLOCK_OC as next power of 2 >= OC
        BLOCK_OC = 1
        while BLOCK_OC < OC:
            BLOCK_OC *= 2

        grid = (N,)
        fused_kernel[grid](
            x, weight, cbias, bias_flat, out,
            N, IC, D, H, W,
            OC, KD, KH, KW,
            OD, OH, OW,
            PD, PH, PW,
            float(self.divisor),
            float(NSPATIAL),
            BLOCK_OC=BLOCK_OC,
            K=K,
            num_warps=4,
        )

        return out.to(x.dtype)