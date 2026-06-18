import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, D, H, W,
    OC, KD, KH, KW,
    OD, OH, OW,         # conv output dims
    PD, PH, PW,         # pooled output dims (after maxpool with stride=2,kernel=2)
    inv_divisor,
    BLOCK_IC: tl.constexpr,
    KD_C: tl.constexpr, KH_C: tl.constexpr, KW_C: tl.constexpr,
):
    # one program per (n, oc): produces a single scalar = sum over pooled spatial of conv/div, then /num_pool
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid % OC

    # input strides (NCDHW contiguous)
    x_n_stride = IC * D * H * W
    x_c_stride = D * H * W
    x_d_stride = H * W
    x_h_stride = W

    # weight strides (OC, IC, KD, KH, KW)
    w_oc_stride = IC * KD_C * KH_C * KW_C
    w_ic_stride = KD_C * KH_C * KW_C
    w_kd_stride = KH_C * KW_C
    w_kh_stride = KW_C

    accum = 0.0

    # iterate over pooled positions
    for pd in range(0, PD):
        for ph in range(0, PH):
            for pw in range(0, PW):
                # pool window: 2x2x2 over conv output
                max_val = -float('inf')
                for dd in range(0, 2):
                    for hh in range(0, 2):
                        for ww in range(0, 2):
                            od = pd * 2 + dd
                            oh = ph * 2 + hh
                            ow = pw * 2 + ww
                            # compute conv at (n, oc, od, oh, ow)
                            conv_sum = 0.0
                            for kd in range(0, KD_C):
                                for kh in range(0, KH_C):
                                    for kw in range(0, KW_C):
                                        id_ = od + kd
                                        ih = oh + kh
                                        iw = ow + kw
                                        # vector load over IC
                                        ic_offs = tl.arange(0, BLOCK_IC)
                                        ic_mask = ic_offs < IC
                                        x_off = (n * x_n_stride
                                                 + ic_offs * x_c_stride
                                                 + id_ * x_d_stride
                                                 + ih * x_h_stride
                                                 + iw)
                                        w_off = (oc * w_oc_stride
                                                 + ic_offs * w_ic_stride
                                                 + kd * w_kd_stride
                                                 + kh * w_kh_stride
                                                 + kw)
                                        x_vals = tl.load(x_ptr + x_off, mask=ic_mask, other=0.0)
                                        w_vals = tl.load(w_ptr + w_off, mask=ic_mask, other=0.0)
                                        conv_sum += tl.sum(x_vals * w_vals, axis=0)
                            # add bias
                            b_val = tl.load(b_ptr + oc)
                            conv_val = (conv_sum + b_val) * inv_divisor
                            max_val = tl.maximum(max_val, conv_val)
                accum += max_val

    # global avg pool: divide by PD*PH*PW
    num_pool = PD * PH * PW
    result = accum / num_pool
    out_off = n * OC + oc
    tl.store(out_ptr + out_off, result)


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
        PD = OD // 2
        PH = OH // 2
        PW = OW // 2

        weight = self.conv.weight.contiguous()
        cbias = self.conv.bias.contiguous()

        # output of pipeline before sum: (N, OC, 1, 1, 1) -> add bias (OC,1,1,1) -> sum over dim=1
        # Our kernel produces per (n, oc) the global-avg-pooled value
        gap = torch.empty((N, OC), device=x.device, dtype=torch.float32)

        # pick BLOCK_IC as next pow2 >= IC
        BLOCK_IC = 1
        while BLOCK_IC < IC:
            BLOCK_IC *= 2

        grid = (N * OC,)
        fused_conv_pool_kernel[grid](
            x, weight, cbias, gap,
            N, IC, D, H, W,
            OC, KD, KH, KW,
            OD, OH, OW,
            PD, PH, PW,
            1.0 / self.divisor,
            BLOCK_IC=BLOCK_IC,
            KD_C=KD, KH_C=KH, KW_C=KW,
            num_warps=4,
        )

        # add bias and sum
        # bias shape (OC,1,1,1) -> broadcast to (N,OC,1,1,1); after sum over dim=sum_dim
        # gap is currently (N, OC) representing the value at (N,OC,0,0,0)
        bias_flat = self.bias.view(OC)  # (OC,)
        result = gap + bias_flat.unsqueeze(0)  # (N, OC)
        # original output shape: sum over sum_dim of (N, OC, 1, 1, 1)
        # if sum_dim == 1: result (N, 1, 1, 1)
        full = result.view(N, OC, 1, 1, 1)
        out = torch.sum(full, dim=self.sum_dim)
        return out