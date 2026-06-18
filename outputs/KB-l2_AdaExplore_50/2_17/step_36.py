import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'IC', 'H_OUT', 'W_OUT'],
)
@triton.jit
def conv2d_im2col_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H_IN, W_IN,
    OC, H_OUT, W_OUT,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # pid_m: OC tile, pid_n: spatial tile, pid_b: batch index
    pid_b = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    HW_OUT = H_OUT * W_OUT
    K = IC * KH * KW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # OC indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # spatial indices

    mask_m = offs_m < OC
    mask_n = offs_n < HW_OUT

    # output spatial coords
    out_h = offs_n // W_OUT
    out_w = offs_n % W_OUT

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K is split into BLOCK_K chunks; each k corresponds to (ic, kh, kw)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # decompose k -> ic, kh, kw
        ic = offs_k // (KH * KW)
        rem = offs_k % (KH * KW)
        kh = rem // KW
        kw = rem % KW

        # Load weight: shape (BLOCK_M, BLOCK_K), w[oc, ic, kh, kw]
        w_off = (offs_m[:, None] * (IC * KH * KW)) + offs_k[None, :]
        w_mask = mask_m[:, None] & mask_k[None, :]
        w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

        # Load input via im2col: x[b, ic, out_h+kh, out_w+kw]
        in_h = out_h[:, None] + kh[None, :]  # (BLOCK_N, BLOCK_K)
        in_w = out_w[:, None] + kw[None, :]
        ic_b = ic[None, :]  # (1, BLOCK_K)

        x_off = (pid_b * IC * H_IN * W_IN
                 + ic_b * (H_IN * W_IN)
                 + in_h * W_IN
                 + in_w)
        x_mask = mask_n[:, None] & mask_k[None, :]
        x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # (BLOCK_N, BLOCK_K)

        # acc[BLOCK_M, BLOCK_N] += w_vals @ x_vals.T
        acc += tl.dot(w_vals, tl.trans(x_vals))

    # bias
    bias = tl.load(b_ptr + offs_m, mask=mask_m, other=0.0)
    acc = acc + bias[:, None]

    # store: out[b, oc, h, w]
    out_off = (pid_b * OC * HW_OUT
               + offs_m[:, None] * HW_OUT
               + offs_n[None, :])
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


def triton_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    N, IC, H_IN, W_IN = x.shape
    OC, _, KH, KW = weight.shape
    H_OUT = H_IN - KH + 1
    W_OUT = W_IN - KW + 1

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    out = torch.empty((N, OC, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(OC, META['BLOCK_M']),
        triton.cdiv(H_OUT * W_OUT, META['BLOCK_N']),
        N,
    )
    conv2d_im2col_kernel[grid](
        x, weight, bias, out,
        N, IC, H_IN, W_IN,
        OC, H_OUT, W_OUT,
        KH, KW,
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 16384}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 32768}, num_warps=16, num_stages=2),
    ],
    key=['HW'],
)
@triton.jit
def instance_norm_div_persistent_kernel(
    x_ptr, out_ptr,
    HW,
    inv_div,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    x_ptr = x_ptr + pid * HW
    out_ptr = out_ptr + pid * HW

    idx = tl.arange(0, BLOCK)
    mask = idx < HW
    v = tl.load(x_ptr + idx, mask=mask, other=0.0)

    sum_x = tl.sum(v, axis=0)
    sum_x2 = tl.sum(v * v, axis=0)

    mean = sum_x / HW
    var = sum_x2 / HW - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    scale = rstd * inv_div
    shift = -mean * scale

    y = v * scale + shift
    tl.store(out_ptr + idx, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=3),
    ],
    key=['HW'],
)
@triton.jit
def instance_norm_div_kernel(
    x_ptr, out_ptr,
    HW,
    inv_div,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    x_ptr = x_ptr + pid * HW
    out_ptr = out_ptr + pid * HW

    sum_x = 0.0
    sum_x2 = 0.0
    for off in range(0, HW, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < HW
        v = tl.load(x_ptr + idx, mask=mask, other=0.0)
        sum_x += tl.sum(v, axis=0)
        sum_x2 += tl.sum(v * v, axis=0)

    mean = sum_x / HW
    var = sum_x2 / HW - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    scale = rstd * inv_div
    shift = -mean * scale

    for off in range(0, HW, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < HW
        v = tl.load(x_ptr + idx, mask=mask, other=0.0)
        y = v * scale + shift
        tl.store(out_ptr + idx, y, mask=mask)


def instance_norm_div(x: torch.Tensor, divide_by: float, eps: float = 1e-5):
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)
    grid = (N * C,)
    if HW <= 32768:
        instance_norm_div_persistent_kernel[grid](
            x, out,
            HW,
            1.0 / divide_by,
            eps,
        )
    else:
        instance_norm_div_kernel[grid](
            x, out,
            HW,
            1.0 / divide_by,
            eps,
        )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.divide_by = float(divide_by)

    def forward(self, x):
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()
        w = self.conv.weight
        b = self.conv.bias
        x = triton_conv2d(x, w, b)
        x = instance_norm_div(x, self.divide_by, eps=1e-5)
        return x