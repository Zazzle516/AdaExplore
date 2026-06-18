import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_transpose_softmax_bias_scale_sigmoid_kernel(
    x_ptr, w_ptr, conv_bias_ptr, bias_ptr, out_ptr,
    N, IC, H, W, OH, OW,
    SCALING: tl.constexpr,
    OC: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    P: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # grid: (N * OH * OW,)
    pid = tl.program_id(0)

    ow = pid % OW
    tmp = pid // OW
    oh = tmp % OH
    n = tmp // OH

    offs_oc = tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # For conv_transpose: out[oh, ow] = sum over ic, kh, kw of
    #   x[n, ic, ih, iw] * w[ic, oc, kh, kw]
    # where ih*S - P + kh = oh  =>  kh = oh + P - ih*S
    # valid kh in [0, K-1]
    # ih_S = oh + P - kh, must be divisible by S and >= 0 and < H*S
    # equivalent: kh in [0,K), ih = (oh + P - kh)/S, requires (oh+P-kh) % S == 0

    # We iterate over kh, kw. For each (kh, kw) compute ih, iw.
    # Then iterate ic.

    for kh in tl.static_range(0, K):
        ih_num = oh + P - kh
        ih = ih_num // S
        ih_valid = (ih_num >= 0) & (ih_num < H * S) & ((ih_num % S) == 0)
        for kw in tl.static_range(0, K):
            iw_num = ow + P - kw
            iw = iw_num // S
            iw_valid = (iw_num >= 0) & (iw_num < W * S) & ((iw_num % S) == 0)
            valid = ih_valid & iw_valid

            if valid:
                # Load x[n, :, ih, iw] for all ic. x is channels_last: (N, C, H, W) stride (C*H*W, 1, C*W, C)
                x_base = n * (IC * H * W) + ih * (IC * W) + iw * IC
                # w shape: (IC, OC, K, K) contiguous; stride (OC*K*K, K*K, K, 1)
                # We want sum over ic of x[ic] * w[ic, oc, kh, kw]
                # For each oc, accumulate over ic.
                for ic in range(0, IC):
                    x_val = tl.load(x_ptr + x_base + ic)
                    w_offs = ic * (OC * K * K) + offs_oc * (K * K) + kh * K + kw
                    w_val = tl.load(w_ptr + w_offs, mask=mask_oc, other=0.0)
                    acc += x_val * w_val

    # add conv bias
    cb = tl.load(conv_bias_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc += cb

    # softmax across OC
    acc_masked = tl.where(mask_oc, acc, -float('inf'))
    max_val = tl.max(acc_masked, axis=0)
    e = tl.exp(acc_masked - max_val)
    e = tl.where(mask_oc, e, 0.0)
    sum_e = tl.sum(e, axis=0)
    sm = e / sum_e

    b = tl.load(bias_ptr + offs_oc, mask=mask_oc, other=0.0)
    y = (sm + b) * SCALING
    out = 1.0 / (1.0 + tl.exp(-y))

    base_out = n * (OC * OH * OW) + oh * (OC * OW) + ow * OC
    tl.store(out_ptr + base_out + offs_oc, out, mask=mask_oc)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = float(scaling_factor)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        N, IC, H, W = x.shape
        K = self.kernel_size
        S = self.stride
        P = self.padding
        OP = self.output_padding
        OH = (H - 1) * S - 2 * P + K + OP
        OW = (W - 1) * S - 2 * P + K + OP
        OC = self.out_channels

        # x in channels_last layout
        x_cl = x.contiguous(memory_format=torch.channels_last)
        # weight: (IC, OC, K, K) - already in this layout for ConvTranspose2d
        w = self.conv_transpose.weight.contiguous()
        cb = self.conv_transpose.bias.contiguous()
        bias_flat = self.bias.view(-1).contiguous()

        # output in channels_last
        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype,
                          memory_format=torch.channels_last)

        BLOCK_OC = triton.next_power_of_2(OC)
        grid = (N * OH * OW,)

        fused_conv_transpose_softmax_bias_scale_sigmoid_kernel[grid](
            x_cl, w, cb, bias_flat, out,
            N, IC, H, W, OH, OW,
            SCALING=self.scaling_factor,
            OC=OC, K=K, S=S, P=P,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
            num_stages=2,
        )
        return out