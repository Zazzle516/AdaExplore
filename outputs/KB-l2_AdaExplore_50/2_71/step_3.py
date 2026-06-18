import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_OC': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 64}, num_warps=4, num_stages=3),
    ],
    key=['N_OUT', 'OC', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv2d_div_lrelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, IC, H, W,
    OC, KH, KW,
    OH, OW,
    N_OUT,  # B * OH * OW
    inv_div,
    neg_slope,
    BLOCK_N: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    mask_n = offs_n < N_OUT
    mask_oc = offs_oc < OC

    # decode n -> (b, oh, ow)
    ow = offs_n % OW
    tmp = offs_n // OW
    oh = tmp % OH
    b = tmp // OH

    acc = tl.zeros((BLOCK_N, BLOCK_OC), dtype=tl.float32)

    # Loop over IC, KH, KW
    for ic in range(0, IC):
        for kh in range(0, KH):
            for kw in range(0, KW):
                ih = oh + kh
                iw = ow + kw
                x_offset = ((b * IC + ic) * H + ih) * W + iw
                x_vals = tl.load(x_ptr + x_offset, mask=mask_n, other=0.0)  # [BLOCK_N]

                w_offset = ((offs_oc * IC + ic) * KH + kh) * KW + kw
                w_vals = tl.load(w_ptr + w_offset, mask=mask_oc, other=0.0)  # [BLOCK_OC]

                acc += x_vals[:, None] * w_vals[None, :]

    # bias
    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + bias[None, :]

    # divide
    acc = acc * inv_div

    # leaky relu
    acc = tl.where(acc >= 0, acc, acc * neg_slope)

    # store: output layout [B, OC, OH, OW]
    out_offset = (b[:, None] * OC + offs_oc[None, :]) * (OH * OW) + (oh[:, None] * OW + ow[:, None])
    mask = mask_n[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_offset, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = float(divisor)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv.weight.contiguous().cuda()
        bias = self.conv.bias.contiguous().cuda()

        B, IC, H, W = x.shape
        OC, _, KH, KW = weight.shape
        OH = H - KH + 1
        OW = W - KW + 1

        out = torch.empty((B, OC, OH, OW), device=x.device, dtype=x.dtype)
        N_OUT = B * OH * OW
        inv_div = 1.0 / self.divisor
        neg_slope = 0.01

        grid = lambda meta: (
            triton.cdiv(N_OUT, meta['BLOCK_N']),
            triton.cdiv(OC, meta['BLOCK_OC']),
        )

        conv2d_div_lrelu_kernel[grid](
            x, weight, bias, out,
            B, IC, H, W,
            OC, KH, KW,
            OH, OW,
            N_OUT,
            inv_div,
            neg_slope,
        )
        return out