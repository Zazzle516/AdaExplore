import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


CONV_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=CONV_CONFIGS, key=['N', 'OC', 'IC', 'H', 'W', 'KH', 'KW'])
@triton.jit
def _conv2d_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W, OC, KH, KW, OH, OW,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # over OC tiles
    pid_n = tl.program_id(1)  # over (N * OH * OW) tiles

    M = OC
    NN = N * OH * OW
    K = IC * KH * KW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    n_idx = offs_n // (OH * OW)
    rem = offs_n % (OH * OW)
    oh_idx = rem // OW
    ow_idx = rem % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k
        k_mask = k_idx < K

        ic = k_idx // (KH * KW)
        kr = k_idx % (KH * KW)
        kh = kr // KW
        kw = kr % KW

        w_offset = (offs_m[:, None] * (IC * KH * KW)
                    + ic[None, :] * (KH * KW)
                    + kh[None, :] * KW
                    + kw[None, :])
        w_mask = (offs_m[:, None] < M) & (k_mask[None, :])
        w_vals = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)

        ic_b = ic[:, None]
        kh_b = kh[:, None]
        kw_b = kw[:, None]
        n_b = n_idx[None, :]
        oh_b = oh_idx[None, :]
        ow_b = ow_idx[None, :]

        h_in2 = oh_b + kh_b
        w_in2 = ow_b + kw_b

        x_offset = (n_b * (IC * H * W)
                    + ic_b * (H * W)
                    + h_in2 * W
                    + w_in2)
        x_mask = (k_mask[:, None]) & (offs_n[None, :] < NN)
        x_vals = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)

        acc += tl.dot(w_vals, x_vals, allow_tf32=False)

    b = tl.load(b_ptr + offs_m, mask=offs_m < M, other=0.0)
    acc = acc + b[:, None]

    # Mish: x * tanh(softplus(x)) -- fused into epilogue
    sp = tl.where(acc > 20.0, acc, tl.log(1.0 + tl.exp(acc)))
    e2 = tl.exp(-2.0 * sp)
    th = (1.0 - e2) / (1.0 + e2)
    acc = acc * th

    # Output layout: [N, OC, OH, OW]
    # offs_n decoded into (n, oh, ow) -> output index = n*OC*OH*OW + oc*OH*OW + oh*OW + ow
    out_offset = (n_idx[None, :] * (OC * OH * OW)
                  + offs_m[:, None] * (OH * OW)
                  + oh_idx[None, :] * OW
                  + ow_idx[None, :])
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < NN)
    tl.store(out_ptr + out_offset, acc, mask=out_mask)


def conv2d_mish_triton(x, weight, bias):
    N, IC, H, W = x.shape
    OC, _, KH, KW = weight.shape
    OH = H - KH + 1
    OW = W - KW + 1

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(OC, meta['BLOCK_M']),
        triton.cdiv(N * OH * OW, meta['BLOCK_N']),
    )

    _conv2d_mish_kernel[grid](
        x, weight, bias, out,
        N, IC, H, W, OC, KH, KW, OH, OW,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        x = x.contiguous()
        if x.is_cuda:
            x = conv2d_mish_triton(x, self.conv.weight, self.conv.bias)
        else:
            x = self.conv(x)
            x = x * torch.tanh(F.softplus(x))
        x = self.bn(x)
        return x