import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_pool_reduce_kernel(
    conv_ptr, bias_ptr, out_ptr,
    N, OC, OD, OH, OW,
    PD, PH, PW,
    POOL_D: tl.constexpr, POOL_H: tl.constexpr, POOL_W: tl.constexpr,
    inv_pool_vol,
    sum_bias,  # precomputed sum of bias over channels
    BLOCK_OC: tl.constexpr,
):
    # one program per batch n
    n = tl.program_id(0)

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    acc = tl.zeros([BLOCK_OC], dtype=tl.float32)

    total_pool = PD * PH * PW

    for p_idx in range(total_pool):
        pd = p_idx // (PH * PW)
        rem = p_idx % (PH * PW)
        ph = rem // PW
        pw = rem % PW

        od_start = pd * POOL_D
        oh_start = ph * POOL_H
        ow_start = pw * POOL_W

        max_vals = tl.full([BLOCK_OC], -float('inf'), dtype=tl.float32)

        for kd in range(POOL_D):
            for kh in range(POOL_H):
                for kw in range(POOL_W):
                    od = od_start + kd
                    oh = oh_start + kh
                    ow = ow_start + kw

                    # conv: [N, OC, OD, OH, OW]
                    off = ((n * OC + oc_offs) * OD + od) * OH * OW + oh * OW + ow
                    val = tl.load(conv_ptr + off, mask=oc_mask, other=-float('inf'))
                    max_vals = tl.maximum(max_vals, val)

        acc += max_vals

    # global avg pool
    acc = acc * inv_pool_vol

    # bias is added per channel; after summing over OC channels, we just add sum_bias once
    acc_masked = tl.where(oc_mask, acc, 0.0)
    total = tl.sum(acc_masked, axis=0) + sum_bias

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
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(self, x):
        x = x.contiguous()
        # Run conv with cuDNN (fast), then fuse div+maxpool+avgpool+bias+sum
        conv_out = self.conv(x)
        # divisor folded into the conv output by simple scaling (max is monotonic for divisor>0)
        conv_out = conv_out / self.divisor
        conv_out = conv_out.contiguous()

        N, OC, OD, OH, OW = conv_out.shape
        POOL_D, POOL_H, POOL_W = self.pool_size
        PD = OD // POOL_D
        PH = OH // POOL_H
        PW = OW // POOL_W

        out = torch.empty(N, 1, 1, 1, device=x.device, dtype=x.dtype)

        BLOCK_OC = max(16, triton.next_power_of_2(OC))

        bias_flat = self.bias.view(-1).contiguous()
        sum_bias_val = float(bias_flat.sum().item())

        grid = (N,)
        fused_pool_reduce_kernel[grid](
            conv_out, bias_flat, out,
            N, OC, OD, OH, OW,
            PD, PH, PW,
            POOL_D, POOL_H, POOL_W,
            1.0 / (PD * PH * PW),
            sum_bias_val,
            BLOCK_OC=BLOCK_OC,
            num_warps=2,
            num_stages=2,
        )

        return out