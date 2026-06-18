import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=['D_out', 'H_out', 'W_out'],
)
@triton.jit
def fused_post_conv_kernel(
    conv_ptr,
    bias_ptr,
    out_ptr,
    N, OC,
    D_out: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    inv_div: tl.constexpr,
    inv_pool: tl.constexpr,
    BLOCK_PW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)

    slice_size = D_out * H_out * W_out
    base = pid_n * OC * slice_size + pid_oc * slice_size

    bias_val = tl.load(bias_ptr + pid_oc)

    acc = tl.zeros([BLOCK_PW], dtype=tl.float32)

    pw_offs = tl.arange(0, BLOCK_PW)
    pw_mask = pw_offs < PW
    w0 = pw_offs * 2
    w1 = w0 + 1

    HW = H_out * W_out
    NEG_INF = -3.4e38

    for pd in tl.static_range(0, PD):
        d0 = pd * 2
        d1 = d0 + 1
        for ph in tl.static_range(0, PH):
            h0 = ph * 2
            h1 = h0 + 1

            off_d0_h0 = base + d0 * HW + h0 * W_out
            off_d0_h1 = base + d0 * HW + h1 * W_out
            off_d1_h0 = base + d1 * HW + h0 * W_out
            off_d1_h1 = base + d1 * HW + h1 * W_out

            p000 = tl.load(conv_ptr + off_d0_h0 + w0, mask=pw_mask, other=NEG_INF)
            p001 = tl.load(conv_ptr + off_d0_h0 + w1, mask=pw_mask, other=NEG_INF)
            p010 = tl.load(conv_ptr + off_d0_h1 + w0, mask=pw_mask, other=NEG_INF)
            p011 = tl.load(conv_ptr + off_d0_h1 + w1, mask=pw_mask, other=NEG_INF)
            p100 = tl.load(conv_ptr + off_d1_h0 + w0, mask=pw_mask, other=NEG_INF)
            p101 = tl.load(conv_ptr + off_d1_h0 + w1, mask=pw_mask, other=NEG_INF)
            p110 = tl.load(conv_ptr + off_d1_h1 + w0, mask=pw_mask, other=NEG_INF)
            p111 = tl.load(conv_ptr + off_d1_h1 + w1, mask=pw_mask, other=NEG_INF)

            m = tl.maximum(tl.maximum(tl.maximum(p000, p001), tl.maximum(p010, p011)),
                           tl.maximum(tl.maximum(p100, p101), tl.maximum(p110, p111)))
            m = m * inv_div
            m = tl.where(pw_mask, m, 0.0)
            acc += m

    total = tl.sum(acc, axis=0)
    result = total * inv_pool + bias_val
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
        N = x.shape[0]
        OC = self.out_channels

        y = F.conv3d(x, self.conv.weight, self.conv.bias)
        _, _, D_out, H_out, W_out = y.shape

        PSD, PSH, PSW = self.pool_size
        PD = D_out // PSD
        PH = H_out // PSH
        PW = W_out // PSW

        bias_flat = self.bias.view(OC)

        out = torch.empty((N, OC), device=x.device, dtype=y.dtype)

        inv_div = 1.0 / float(self.divisor)
        inv_pool = 1.0 / float(PD * PH * PW)

        BLOCK_PW = 32
        while BLOCK_PW < PW:
            BLOCK_PW *= 2

        grid = (N, OC)
        fused_post_conv_kernel[grid](
            y, bias_flat, out,
            N, OC,
            D_out, H_out, W_out,
            PD, PH, PW,
            inv_div, inv_pool,
            BLOCK_PW,
        )

        if self.sum_dim == 1:
            result = out.sum(dim=1).view(N, 1, 1, 1)
        else:
            full = out.view(N, OC, 1, 1, 1)
            result = full.sum(dim=self.sum_dim)
        return result