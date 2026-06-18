import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
    ],
    key=['IC', 'OC', 'H_OUT', 'W_OUT', 'KH', 'KW'],
)
@triton.jit
def conv2d_div_lrelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, H_IN, W_IN,
    OC: tl.constexpr, H_OUT, W_OUT,
    KH: tl.constexpr, KW: tl.constexpr,
    K: tl.constexpr, K_PAD: tl.constexpr,
    neg_slope: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, K_PAD)

    M = H_OUT * W_OUT
    mask_m = offs_m < M
    mask_n = offs_n < OC
    mask_k = offs_k < K

    out_h = offs_m // W_OUT
    out_w = offs_m % W_OUT

    KWIC = KW * IC
    k_kh = offs_k // KWIC
    k_rem = offs_k % KWIC
    k_kw = k_rem // IC
    k_ic = k_rem % IC

    in_h = out_h[:, None] + k_kh[None, :]
    in_w = out_w[:, None] + k_kw[None, :]
    x_off = (pid_b * H_IN * W_IN * IC
             + in_h * (W_IN * IC)
             + in_w * IC
             + k_ic[None, :])
    x_mask = mask_m[:, None] & mask_k[None, :]
    x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

    w_off = offs_k[:, None] * OC + offs_n[None, :]
    w_mask = mask_k[:, None] & mask_n[None, :]
    w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

    acc = tl.dot(x_vals, w_vals)

    b_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b_vals[None, :]
    acc = tl.where(acc >= 0, acc, acc * neg_slope)

    out_off = (pid_b * OC * M
               + offs_n[None, :] * M
               + offs_m[:, None])
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self._cached_w = None
        self._cached_b = None

    def _prepare(self, device):
        inv_divisor = 1.0 / float(self.divisor)
        w = self.conv.weight.detach().to(device)
        w = w * inv_divisor
        w = w.permute(2, 3, 1, 0).contiguous()
        OC = self.out_channels
        K = self.in_channels * self.kernel_size * self.kernel_size
        w = w.view(K, OC).contiguous()
        b = self.conv.bias.detach().to(device) * inv_divisor
        b = b.contiguous()
        self._cached_w = w
        self._cached_b = b

    def forward(self, x):
        x = x.contiguous().cuda()
        if self._cached_w is None or self._cached_w.device != x.device:
            self._prepare(x.device)

        N, IC, H_IN, W_IN = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        H_OUT = H_IN - KH + 1
        W_OUT = W_IN - KW + 1
        K = IC * KH * KW
        K_PAD = _next_pow2(max(K, 16))

        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        out = torch.empty((N, OC, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

        neg_slope = 0.01

        grid = lambda meta: (
            N,
            triton.cdiv(H_OUT * W_OUT, meta['BLOCK_M']),
            triton.cdiv(OC, meta['BLOCK_N']),
        )

        conv2d_div_lrelu_kernel[grid](
            x_nhwc, self._cached_w, self._cached_b, out,
            N, IC, H_IN, W_IN,
            OC, H_OUT, W_OUT,
            KH, KW,
            K, K_PAD,
            neg_slope,
        )
        return out