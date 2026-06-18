import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Fused gather-form ConvTranspose3d + pool(2)+pool(3) + sum over channels.
# Output spatial after conv_transpose with input 32, k=5, s=2, p=2 is 63 (= (32-1)*2 - 2*2 + 5).
# After max_pool(k=2): 31. After max_pool(k=3): 10. (Strides equal kernel sizes; floor.)
# Effective window per final output element = 6x6x6 (in conv_transpose output coordinates),
# starting at (6*od, 6*oh, 6*ow) and going up to <(6*od+6, 6*oh+6, 6*ow+6).
#
# For each output voxel (od, oh, ow) and each output channel oc, we compute the
# conv_transpose output value at all 216 positions on the fly, take max, then sum over oc.
#
# conv_transpose3d formula (stride=s, padding=p, kernel K):
#   y[n, oc, dz, dy, dx] = bias[oc] + sum_{ic, kd, kh, kw}
#       x[n, ic, (dz + p - kd)/s, (dy + p - kh)/s, (dx + p - kw)/s] * w[ic, oc, kd, kh, kw]
#   where the divisions must be exact (i.e., (dz+p-kd) % s == 0) and indices in range.

@triton.jit
def fused_convt_pool_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, OC,
    ID, IH, IW,
    OD_full, OH_full, OW_full,  # conv_transpose output spatial size
    OD, OH, OW,                  # final pooled output spatial size
    K: tl.constexpr,
    S: tl.constexpr,
    P: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # grid: (N * OD * OH * OW,)
    pid = tl.program_id(0)
    ow = pid % OW
    tmp = pid // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    n = tmp // OD

    d_start = od * 6
    h_start = oh * 6
    w_start = ow * 6

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Load bias for all OC
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    # max accumulator per oc
    neg_inf = float('-inf')
    max_vals = tl.full((BLOCK_OC,), neg_inf, dtype=tl.float32)

    # input strides (contiguous): x is (N, IC, ID, IH, IW)
    # weight is (IC, OC, K, K, K)

    # Iterate over the 6x6x6 spatial window
    for dd in tl.static_range(0, 6):
        dz = d_start + dd
        for hh in tl.static_range(0, 6):
            dy = h_start + hh
            for ww in tl.static_range(0, 6):
                dx = w_start + ww

                # Initialize accumulator with bias
                acc = bias  # shape [BLOCK_OC]

                # For each kernel position, determine which input voxel contributes
                # (dz + P - kd) must be divisible by S, then iz = (dz + P - kd) / S
                for kd in tl.static_range(0, K):
                    num_d = dz + P - kd
                    valid_d = (num_d >= 0) & ((num_d % S) == 0)
                    iz = num_d // S
                    valid_d = valid_d & (iz >= 0) & (iz < ID)

                    for kh in tl.static_range(0, K):
                        num_h = dy + P - kh
                        valid_h = (num_h >= 0) & ((num_h % S) == 0)
                        iy = num_h // S
                        valid_h = valid_h & (iy >= 0) & (iy < IH)

                        for kw in tl.static_range(0, K):
                            num_w = dx + P - kw
                            valid_w = (num_w >= 0) & ((num_w % S) == 0)
                            ix = num_w // S
                            valid_w = valid_w & (ix >= 0) & (ix < IW)

                            valid = valid_d & valid_h & valid_w

                            if valid:
                                # Sum over IC: dot product of x[n, :, iz, iy, ix] and w[:, :, kd, kh, kw]
                                # Iterate over IC in chunks of BLOCK_IC
                                for ic_start in tl.static_range(0, IC, BLOCK_IC):
                                    ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                                    ic_mask = ic_offs < IC
                                    # x[n, ic, iz, iy, ix]
                                    x_off = ((((n * IC) + ic_offs) * ID + iz) * IH + iy) * IW + ix
                                    x_vals = tl.load(x_ptr + x_off, mask=ic_mask, other=0.0)  # [BLOCK_IC]

                                    # w[ic, oc, kd, kh, kw]: shape (IC, OC, K, K, K)
                                    # offset: ((((ic * OC) + oc) * K + kd) * K + kh) * K + kw
                                    w_off = ((((ic_offs[:, None] * OC) + oc_offs[None, :]) * K + kd) * K + kh) * K + kw
                                    w_mask = ic_mask[:, None] & oc_mask[None, :]
                                    w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # [BLOCK_IC, BLOCK_OC]

                                    # acc += sum_ic x_vals[ic] * w_vals[ic, oc]
                                    acc = acc + tl.sum(x_vals[:, None] * w_vals, axis=0)

                # Update max
                acc = tl.where(oc_mask, acc, neg_inf)
                max_vals = tl.maximum(max_vals, acc)

    # Sum over OC
    s = tl.sum(tl.where(oc_mask, max_vals, 0.0), axis=0)

    out_off = ((n * OD + od) * OH + oh) * OW + ow
    tl.store(out_ptr + out_off, s)


def fused_convt_pool_sum(x, weight, bias, stride, padding, kernel_size):
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    N, IC, ID, IH, IW = x.shape
    IC_w, OC, K1, K2, K3 = weight.shape
    assert IC_w == IC and K1 == K2 == K3 == kernel_size

    # ConvTranspose3d output spatial size
    OD_full = (ID - 1) * stride - 2 * padding + kernel_size
    OH_full = (IH - 1) * stride - 2 * padding + kernel_size
    OW_full = (IW - 1) * stride - 2 * padding + kernel_size

    OD = OD_full // 6
    OH = OH_full // 6
    OW = OW_full // 6

    out = torch.empty((N, 1, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = triton.next_power_of_2(OC)
    BLOCK_IC = triton.next_power_of_2(IC)

    grid = (N * OD * OH * OW,)
    fused_convt_pool_sum_kernel[grid](
        x, weight, bias, out,
        N, IC, OC,
        ID, IH, IW,
        OD_full, OH_full, OW_full,
        OD, OH, OW,
        K=kernel_size,
        S=stride,
        P=padding,
        BLOCK_OC=BLOCK_OC,
        BLOCK_IC=BLOCK_IC,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return fused_convt_pool_sum(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.stride,
            self.padding,
            self.kernel_size,
        )