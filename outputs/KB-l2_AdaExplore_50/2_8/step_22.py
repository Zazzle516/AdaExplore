import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_kernel(
    x_ptr, w_ptr, b_conv_ptr, bias_ptr, out_ptr,
    N, IC, D, H, W,
    OC, KD, KH, KW,
    OD, OH, OW,  # conv output dims
    PD, PH, PW,  # pooled dims (after maxpool 2x2x2)
    divisor,
    BLOCK_OC: tl.constexpr,
    K_DIM: tl.constexpr,
):
    # one program per n
    n = tl.program_id(0)

    # K_DIM = IC * KD * KH * KW
    # Load full weight matrix [OC, K_DIM] once. OC=16, K_DIM=216 => 13.5KB
    oc_offs = tl.arange(0, BLOCK_OC)
    k_offs = tl.arange(0, K_DIM)
    w_mask = (oc_offs[:, None] < OC) & (k_offs[None, :] < K_DIM)
    w = tl.load(w_ptr + oc_offs[:, None] * K_DIM + k_offs[None, :], mask=w_mask, other=0.0)
    # conv bias
    bconv = tl.load(b_conv_ptr + oc_offs, mask=oc_offs < OC, other=0.0)

    # running sum per oc
    acc_sum = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # iterate over pooled output positions
    # for each (pd, ph, pw) in pooled grid:
    #   - 2x2x2 = 8 conv positions (od, oh, ow) where od in [2pd, 2pd+1], etc.
    #   - for each, compute conv outputs for all OC via gemv:
    #       y[oc] = sum_{ic, kd, kh, kw} x[n, ic, od+kd, oh+kh, ow+kw] * w[oc, ic, kd, kh, kw] + bconv[oc]
    #   - divide by divisor, then take max over 8 positions per oc
    #   - accumulate to per-oc sum (for mean)
    # After loop: mean = sum / (PD*PH*PW), add bias, sum over oc, store

    NSPATIAL = PD * PH * PW

    for pd in range(0, PD):
        for ph in range(0, PH):
            for pw in range(0, PW):
                # max across 8 conv outputs per oc
                max_val = tl.full((BLOCK_OC,), -float('inf'), dtype=tl.float32)
                for ddz in range(0, 2):
                    for ddy in range(0, 2):
                        for ddx in range(0, 2):
                            od = pd * 2 + ddz
                            oh = ph * 2 + ddy
                            ow = pw * 2 + ddx
                            # compute x patch of shape [K_DIM]
                            # k = ic*KD*KH*KW + kd*KH*KW + kh*KW + kw
                            ic_idx = k_offs // (KD * KH * KW)
                            rem = k_offs % (KD * KH * KW)
                            kd_idx = rem // (KH * KW)
                            rem2 = rem % (KH * KW)
                            kh_idx = rem2 // KW
                            kw_idx = rem2 % KW
                            d_idx = od + kd_idx
                            h_idx = oh + kh_idx
                            w_idx = ow + kw_idx
                            x_off = (n * IC * D * H * W
                                     + ic_idx * D * H * W
                                     + d_idx * H * W
                                     + h_idx * W
                                     + w_idx)
                            x_mask = (ic_idx < IC) & (d_idx < D) & (h_idx < H) & (w_idx < W)
                            x_vec = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)
                            # conv: y[oc] = sum_k w[oc, k] * x_vec[k]
                            y = tl.sum(w * x_vec[None, :], axis=1) + bconv
                            y = y / divisor
                            max_val = tl.maximum(max_val, y)
                acc_sum = acc_sum + max_val

    # mean across spatial
    mean_val = acc_sum / NSPATIAL
    # add bias
    bias_vals = tl.load(bias_ptr + oc_offs, mask=oc_offs < OC, other=0.0)
    final = mean_val + bias_vals
    # mask out invalid oc
    final = tl.where(oc_offs < OC, final, 0.0)
    # sum over oc
    out_val = tl.sum(final, axis=0)
    tl.store(out_ptr + n, out_val)


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

        K_DIM = IC * KD * KH * KW  # 8*3*3*3 = 216

        # weight shape [OC, IC, KD, KH, KW] -> [OC, K_DIM]
        w = self.conv.weight.contiguous().view(OC, K_DIM)
        bconv = self.conv.bias.contiguous()
        bias_flat = self.bias.contiguous().view(-1)

        out = torch.empty(N, device=x.device, dtype=x.dtype)

        # pad K_DIM and OC to next power of 2 for triton
        BLOCK_OC = 16
        # K_DIM = 216, pad to 256
        K_PAD = 256

        # pad weight tensor along K dimension
        w_padded = torch.zeros(BLOCK_OC, K_PAD, device=x.device, dtype=x.dtype)
        w_padded[:OC, :K_DIM] = w

        grid = (N,)
        fused_kernel[grid](
            x, w_padded, bconv, bias_flat, out,
            N, IC, D, H, W,
            OC, KD, KH, KW,
            OD, OH, OW,
            PD, PH, PW,
            float(self.divisor),
            BLOCK_OC=BLOCK_OC,
            K_DIM=K_PAD,
            num_warps=4,
        )

        return out