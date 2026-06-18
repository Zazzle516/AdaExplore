import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _mish_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # softplus = log(1 + exp(x)), numerically stable
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    # tanh via exp
    e2 = tl.exp(-2.0 * sp)
    t = (1.0 - e2) / (1.0 + e2)
    out = x * t
    tl.store(out_ptr + offsets, out, mask=mask)


def mish_triton(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: ((n + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _mish_kernel[grid](x, out, n, BLOCK_SIZE=1024)
    return out


CONV_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=CONV_CONFIGS, key=['N', 'OC', 'IC', 'H', 'W', 'KH', 'KW'])
@triton.jit
def _conv2d_fused_kernel(
    x_ptr, w_ptr, b_ptr, scale_ptr, shift_ptr, out_ptr,
    N, IC, H, W, OC, KH, KW, OH, OW,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # over OC tiles
    pid_n = tl.program_id(1)  # over (N * OH * OW) tiles

    M = OC
    NN = N * OH * OW
    K = IC * KH * KW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # OC indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # spatial indices

    # decode n, oh, ow from offs_n
    n_idx = offs_n // (OH * OW)
    rem = offs_n % (OH * OW)
    oh_idx = rem // OW
    ow_idx = rem % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]
        k_mask = k_idx < K

        # decode k -> ic, kh, kw
        ic = k_idx // (KH * KW)
        kr = k_idx % (KH * KW)
        kh = kr // KW
        kw = kr % KW

        # weight: [OC, IC, KH, KW]; load w[offs_m, ic, kh, kw] -> [BLOCK_M, BLOCK_K]
        w_offset = (offs_m[:, None] * (IC * KH * KW)
                    + ic[None, :] * (KH * KW)
                    + kh[None, :] * KW
                    + kw[None, :])
        w_mask = (offs_m[:, None] < M) & (k_mask[None, :])
        w_vals = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)

        # input position
        h_in = oh_idx[:, None] + kh[None, :]  # [BLOCK_N, BLOCK_K]... wait we want [BLOCK_K, BLOCK_N]
        # Recompute properly: we want x[n, ic, h_in, w_in] indexed by (k, n_spatial) -> shape [BLOCK_K, BLOCK_N]
        # Use offs_n along columns
        ic_b = ic[:, None]                # [BLOCK_K, 1]
        kh_b = kh[:, None]
        kw_b = kw[:, None]
        n_b = n_idx[None, :]              # [1, BLOCK_N]
        oh_b = oh_idx[None, :]
        ow_b = ow_idx[None, :]

        h_in2 = oh_b + kh_b               # [BLOCK_K, BLOCK_N]
        w_in2 = ow_b + kw_b

        x_offset = (n_b * (IC * H * W)
                    + ic_b * (H * W)
                    + h_in2 * W
                    + w_in2)
        x_mask = (k_mask[:, None]) & (offs_n[None, :] < NN)
        x_vals = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)

        acc += tl.dot(w_vals, x_vals, allow_tf32=False)

    # bias
    b = tl.load(b_ptr + offs_m, mask=offs_m < M, other=0.0)
    acc = acc + b[:, None]

    # apply BN scale/shift fused (eval time, BN folded)
    scale = tl.load(scale_ptr + offs_m, mask=offs_m < M, other=0.0)
    shift = tl.load(shift_ptr + offs_m, mask=offs_m < M, other=0.0)

    # We do NOT apply mish here because BN must operate on conv output, then mish? 
    # Wait, the original order: y = mish(conv(x)); then bn(y). So mish happens BEFORE bn.
    # For training BN this would be runtime stats. Here we just do conv+bias and let separate steps handle.
    # So this kernel just outputs conv (no scale/shift). Let's ignore scale/shift here.
    out_offset = offs_m[:, None] * NN + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < NN)
    tl.store(out_ptr + out_offset, acc, mask=out_mask)


def conv2d_triton(x, weight, bias):
    """
    x: [N, IC, H, W], weight: [OC, IC, KH, KW], bias: [OC]
    returns out laid out as [OC, N*OH*OW] for easy further processing,
    but we'll reshape to [N, OC, OH, OW].
    """
    N, IC, H, W = x.shape
    OC, _, KH, KW = weight.shape
    OH = H - KH + 1
    OW = W - KW + 1

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # buffer in [OC, N*OH*OW]
    out_buf = torch.empty((OC, N * OH * OW), device=x.device, dtype=x.dtype)

    # dummy scale/shift (unused)
    dummy = torch.empty(OC, device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(OC, meta['BLOCK_M']),
        triton.cdiv(N * OH * OW, meta['BLOCK_N']),
    )

    _conv2d_fused_kernel[grid](
        x, weight, bias, dummy, dummy, out_buf,
        N, IC, H, W, OC, KH, KW, OH, OW,
    )

    # reshape to [N, OC, OH, OW]
    out = out_buf.view(OC, N, OH, OW).permute(1, 0, 2, 3).contiguous()
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        x = x.contiguous()
        if x.is_cuda:
            x = conv2d_triton(x, self.conv.weight, self.conv.bias)
        else:
            x = self.conv(x)
        # mish: x * tanh(softplus(x))
        if x.is_cuda:
            x = mish_triton(x)
        else:
            x = x * torch.tanh(F.softplus(x))
        x = self.bn(x)
        return x