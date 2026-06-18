import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Fused kernel: for each (n, oc, od2, oh2, ow2) compute the 4x4x4 block of conv-transpose
# outputs and average them, then apply BN scale+shift. This avoids materializing the
# full conv-transpose output.
#
# Output spatial sizes (with padding=1, stride=2, kernel=3, ID=IH=IW=32):
#   OD = (ID-1)*2 - 2 + 3 = 63
# After avg_pool 2 then 2, effective avg_pool 4, output = 63//4 = 15, with the final
# avg_pool taking pairs from the 4-pool... Actually torch's avg_pool with kernel=2 on
# odd input drops the last element (floor). Composing avg_pool(2) twice:
#   intermediate size = 63 // 2 = 31
#   final size = 31 // 2 = 15
# This is NOT equivalent to avg_pool(4) on the original 63 (which would give 15 too,
# but with different windows: floor(63/4)=15, windows at [0..3],[4..7],...,[56..59]).
# Both methods give 15, but the windows are identical: avg_pool(2) twice with no
# padding gives windows of size 4 starting at 0,4,8,... so they are equivalent.
#
# Actually avg_pool(2) on 63 -> size 31, windows [0,1],[2,3],...,[60,61] (drops idx 62)
# Then avg_pool(2) on 31 -> size 15, windows over intermediate [0,1],[2,3],...,[28,29]
# Each final cell = avg of 4 consecutive original cells starting at 0,4,8,...,56.
# So it is equivalent to a strided 4x4x4 average pool starting at 0 with stride 4.

@triton.jit
def fused_convt_bn_pool_kernel(
    x_ptr, w_ptr, b_ptr, scale_ptr, shift_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    OD2, OH2, OW2,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    IC_C: tl.constexpr,
):
    # One program per (n, oc, od2 * OH2 * OW2 + oh2 * OW2 + ow2)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    ow2 = pid_sp % OW2
    tmp = pid_sp // OW2
    oh2 = tmp % OH2
    od2 = tmp // OH2

    od_base = od2 * 4
    oh_base = oh2 * 4
    ow_base = ow2 * 4

    acc = 0.0

    # Loop over the 4x4x4 output window
    for dd in tl.static_range(4):
        od = od_base + dd
        for hh in tl.static_range(4):
            oh = oh_base + hh
            for ww in tl.static_range(4):
                ow = ow_base + ww

                # Compute conv-transpose output at (n, oc, od, oh, ow)
                # = sum over (ic, kd, kh, kw) of x[n, ic, id, ih, iw] * w[ic, oc, kd, kh, kw]
                # where id = (od + PD - kd) / SD, valid when divisible and in [0, ID)
                val = 0.0
                for kd in tl.static_range(KD):
                    id_num = od + PD - kd
                    id_val = id_num // SD
                    id_valid = (id_num >= 0) & ((id_num - id_val * SD) == 0) & (id_val >= 0) & (id_val < ID)
                    for kh in tl.static_range(KH):
                        ih_num = oh + PH - kh
                        ih_val = ih_num // SH
                        ih_valid = (ih_num >= 0) & ((ih_num - ih_val * SH) == 0) & (ih_val >= 0) & (ih_val < IH)
                        for kw in tl.static_range(KW):
                            iw_num = ow + PW - kw
                            iw_val = iw_num // SW
                            iw_valid = (iw_num >= 0) & ((iw_num - iw_val * SW) == 0) & (iw_val >= 0) & (iw_val < IW)
                            valid = id_valid & ih_valid & iw_valid

                            # Load x[n, :, id_val, ih_val, iw_val] for all ic, dot with w[:, oc, kd, kh, kw]
                            ic_offs = tl.arange(0, IC_C)
                            ic_mask = ic_offs < IC

                            x_off = ((pid_n * IC + ic_offs) * ID + id_val) * IH * IW + ih_val * IW + iw_val
                            x_vals = tl.load(x_ptr + x_off, mask=ic_mask & valid, other=0.0)

                            w_off = ((ic_offs * OC + pid_oc) * KD + kd) * KH * KW + kh * KW + kw
                            w_vals = tl.load(w_ptr + w_off, mask=ic_mask, other=0.0)

                            val += tl.sum(x_vals * w_vals, axis=0)

                acc += val

    # Add bias (conv-transpose bias)
    b_val = tl.load(b_ptr + pid_oc)
    acc = acc / 64.0 + b_val

    # Apply BN: out = acc * scale + shift  (because avg is linear; scale*bias absorbed if we
    # do (acc+bias)*scale + shift). But here we add the bias then apply BN.
    scale = tl.load(scale_ptr + pid_oc)
    shift = tl.load(shift_ptr + pid_oc)
    out = acc * scale + shift

    out_off = ((pid_n * OC + pid_oc) * OD2 + od2) * OH2 * OW2 + oh2 * OW2 + ow2
    tl.store(out_ptr + out_off, out)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous()
        bias = self.conv_transpose.bias.contiguous()

        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        SD = SH = SW = self.stride
        PD = PH = PW = self.padding

        OD = (ID - 1) * SD - 2 * PD + KD
        OH = (IH - 1) * SH - 2 * PH + KH
        OW = (IW - 1) * SW - 2 * PW + KW

        OD2 = OD // 4
        OH2 = OH // 4
        OW2 = OW // 4

        # Fallback for training mode
        if self.training or OD2 * 4 == 0 or OH2 * 4 == 0 or OW2 * 4 == 0:
            x_ = self.conv_transpose(x)
            x_ = self.batch_norm(x_)
            x_ = F.avg_pool3d(x_, 2)
            x_ = F.avg_pool3d(x_, 2)
            return x_

        running_mean = self.batch_norm.running_mean
        running_var = self.batch_norm.running_var
        gamma = self.batch_norm.weight
        beta = self.batch_norm.bias
        eps = self.batch_norm.eps

        inv_std = torch.rsqrt(running_var + eps)
        scale = (gamma * inv_std).contiguous()
        shift = (beta - running_mean * scale).contiguous()

        out = torch.empty((N, OC, OD2, OH2, OW2), device=x.device, dtype=x.dtype)

        # Pick IC_C as next pow2 >= IC
        IC_C = 1
        while IC_C < IC:
            IC_C *= 2
        if IC_C < 4:
            IC_C = 4

        grid = (N, OC, OD2 * OH2 * OW2)

        fused_convt_bn_pool_kernel[grid](
            x, weight, bias, scale, shift, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            OD2, OH2, OW2,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            IC_C=IC_C,
            num_warps=2,
            num_stages=2,
        )

        return out