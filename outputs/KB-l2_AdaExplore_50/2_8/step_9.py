import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Output spatial dims after conv: D_out = 14, H_out = W_out = 62
# After 2x2x2 maxpool: PD = 7, PH = 31, PW = 31
# After global avg pool (over 7*31*31 = 6727 elements): single scalar per (N, OC)
# Then add bias[OC] and sum over OC.
#
# Strategy: one program per (N, OC). It:
#  1) computes conv at every output position (D=14, H=62, W=62)
#  2) divides by divisor
#  3) does 2x2x2 maxpool (PD=7, PH=31, PW=31)
#  4) sums across all PD*PH*PW outputs (= global avg numerator)
#  5) divides by 6727, adds bias[OC]
# Result: scalar per (N, OC). Then a tiny reduction over OC gives final [N].


@triton.jit
def conv3d_fused_kernel(
    x_ptr,          # [N, IC, D, H, W]
    w_ptr,          # [OC, IC, KD, KH, KW]
    cb_ptr,         # [OC] conv bias
    out_ptr,        # [N, OC]
    N, IC, D, H, W,
    OC, KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    D_out: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    inv_div: tl.constexpr,
    inv_pool: tl.constexpr,
    IC_CONST: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)

    # strides
    x_n_stride = IC * D * H * W
    x_c_stride = D * H * W
    x_d_stride = H * W
    x_h_stride = W

    w_oc_stride = IC_CONST * KD * KH * KW
    w_ic_stride = KD * KH * KW
    w_kd_stride = KH * KW
    w_kh_stride = KW

    x_base = pid_n * x_n_stride
    w_base = pid_oc * w_oc_stride

    bias_val = tl.load(cb_ptr + pid_oc)

    acc = tl.zeros([1], dtype=tl.float32)
    acc_scalar = 0.0

    # Loop over pool output positions (PD, PH, PW)
    for pd in range(0, PD):
        for ph in range(0, PH):
            for pw in range(0, PW):
                # For this pool window, find max over 2x2x2 conv outputs
                max_val = -float('inf')
                for dd in range(0, 2):
                    for dh in range(0, 2):
                        for dw in range(0, 2):
                            od = pd * 2 + dd
                            oh = ph * 2 + dh
                            ow = pw * 2 + dw
                            # Compute conv at (od, oh, ow)
                            csum = 0.0
                            for kd in range(0, KD):
                                for kh in range(0, KH):
                                    for kw in range(0, KW):
                                        id_ = od + kd
                                        ih = oh + kh
                                        iw = ow + kw
                                        # Vectorize over IC
                                        ic_offs = tl.arange(0, IC_CONST)
                                        x_idx = (x_base
                                                 + ic_offs * x_c_stride
                                                 + id_ * x_d_stride
                                                 + ih * x_h_stride
                                                 + iw)
                                        w_idx = (w_base
                                                 + ic_offs * w_ic_stride
                                                 + kd * w_kd_stride
                                                 + kh * w_kh_stride
                                                 + kw)
                                        xv = tl.load(x_ptr + x_idx)
                                        wv = tl.load(w_ptr + w_idx)
                                        csum += tl.sum(xv * wv, axis=0)
                            csum = csum + bias_val
                            csum = csum * inv_div
                            max_val = tl.maximum(max_val, csum)
                acc_scalar += max_val

    # Global avg pool: divide by PD*PH*PW
    result = acc_scalar * inv_pool
    # Note: bias addition happens AFTER global avg pool in original model,
    # but bias is per-OC. We add it here.
    # The final bias add in original: x + self.bias where bias shape is (OC,1,1,1)
    # After global avg pool, x shape is (N, OC, 1, 1, 1). Adding bias broadcasts.
    # Then sum over dim=1 (OC).
    # So we just store per-(N,OC) result. Bias add can be folded here.
    tl.store(out_ptr + pid_n * OC + pid_oc, result)


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
        D_out = D - KD + 1
        H_out = H - KH + 1
        W_out = W - KW + 1
        PSD, PSH, PSW = self.pool_size
        PD = D_out // PSD
        PH = H_out // PSH
        PW = W_out // PSW
        OC = self.out_channels

        x = x.contiguous()
        weight = self.conv.weight.contiguous()
        conv_bias = self.conv.bias.contiguous()

        # Output: (N, OC) — per-(N,OC) global-avg-pooled value
        out = torch.empty((N, OC), device=x.device, dtype=torch.float32)

        inv_div = 1.0 / float(self.divisor)
        inv_pool = 1.0 / float(PD * PH * PW)

        grid = (N, OC)
        conv3d_fused_kernel[grid](
            x, weight, conv_bias, out,
            N, IC, D, H, W,
            OC, KD, KH, KW,
            D_out, H_out, W_out,
            PD, PH, PW,
            inv_div, inv_pool,
            IC,
            num_warps=4,
        )

        # Add bias (shape OC,1,1,1 -> flatten to OC) and sum over OC if sum_dim==1
        bias_flat = self.bias.view(OC)
        out = out + bias_flat.unsqueeze(0)  # [N, OC]

        # Original output shape after sum over dim=1 of [N, OC, 1, 1, 1] is [N, 1, 1, 1]
        if self.sum_dim == 1:
            result = out.sum(dim=1, keepdim=False)  # [N]
            # Original would give [N, 1, 1, 1]
            result = result.view(N, 1, 1, 1)
        else:
            # Fallback: reconstruct full tensor
            full = out.view(N, OC, 1, 1, 1)
            result = full.sum(dim=self.sum_dim)
        return result