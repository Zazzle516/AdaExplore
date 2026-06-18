import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 16}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 16}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 16}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['OC', 'OUT_HW', 'IC_KH_KW'])
@triton.jit
def conv2d_relu_bias_kernel(
    x_ptr, w_ptr, b_ptr, bias_ptr, out_ptr,
    N, IC, H, W,
    OC, KH, KW,
    OH, OW,
    OUT_HW, IC_KH_KW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_sp = tl.program_id(0)    # output spatial tile (fast-varying)
    pid_n_oc = tl.program_id(1)  # batch * (OC tiles)

    num_sp_tiles = tl.cdiv(OUT_HW, BLOCK_M)
    num_oc_tiles = tl.cdiv(OC, BLOCK_N)
    n_idx = pid_n_oc // num_oc_tiles
    oc_tile = pid_n_oc % num_oc_tiles

    # L2 swizzle on spatial dimension for better weight reuse across (N,OC)
    group_id = pid_sp // GROUP_M
    first_in_group = group_id * GROUP_M
    group_size = min(num_sp_tiles - first_in_group, GROUP_M)
    sp_tile = first_in_group + ((pid_sp % GROUP_M) % group_size)

    sp_start = sp_tile * BLOCK_M
    oc_start = oc_tile * BLOCK_N

    offs_m = sp_start + tl.arange(0, BLOCK_M)  # output spatial positions
    offs_n = oc_start + tl.arange(0, BLOCK_N)  # output channels
    offs_k = tl.arange(0, BLOCK_K)

    oh = offs_m // OW
    ow = offs_m % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, IC_KH_KW, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]
        ic = k_idx // (KH * KW)
        rem = k_idx % (KH * KW)
        kh = rem // KW
        kw = rem % KW

        # Input addresses: x[n, ic, oh+kh, ow+kw]
        ih = oh[:, None] + kh[None, :]  # [BLOCK_M, BLOCK_K]
        iw = ow[:, None] + kw[None, :]
        x_addr = (n_idx * stride_xn
                  + ic[None, :] * stride_xc
                  + ih * stride_xh
                  + iw * stride_xw)
        x_mask = (offs_m[:, None] < OUT_HW) & (k_idx[None, :] < IC_KH_KW)
        x_vals = tl.load(x_ptr + x_addr, mask=x_mask, other=0.0)

        # Weight addresses: w[oc, ic, kh, kw]
        w_addr = (offs_n[None, :] * stride_wo
                  + ic[:, None] * stride_wi
                  + kh[:, None] * stride_wkh
                  + kw[:, None] * stride_wkw)
        w_mask = (k_idx[:, None] < IC_KH_KW) & (offs_n[None, :] < OC)
        w_vals = tl.load(w_ptr + w_addr, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    # Add conv bias
    b_vals = tl.load(b_ptr + offs_n, mask=offs_n < OC, other=0.0)
    acc += b_vals[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # Add extra fused bias (per output channel)
    extra_b = tl.load(bias_ptr + offs_n, mask=offs_n < OC, other=0.0)
    acc += extra_b[None, :]

    # Store
    out_addr = (n_idx * stride_on
                + offs_n[None, :] * stride_oc
                + oh[:, None] * stride_oh
                + ow[:, None] * stride_ow)
    out_mask = (offs_m[:, None] < OUT_HW) & (offs_n[None, :] < OC)
    tl.store(out_ptr + out_addr, acc, mask=out_mask)


def conv2d_relu_bias(x, w, b, bias):
    assert x.is_cuda and w.is_cuda
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()
    bias_flat = bias.contiguous().view(-1)

    N, IC, H, W = x.shape
    OC, _, KH, KW = w.shape
    OH = H - KH + 1
    OW = W - KW + 1

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    OUT_HW = OH * OW
    IC_KH_KW = IC * KH * KW

    grid = lambda meta: (
        triton.cdiv(OUT_HW, meta["BLOCK_M"]),
        N * triton.cdiv(OC, meta["BLOCK_N"]),
    )

    conv2d_relu_bias_kernel[grid](
        x, w, b, bias_flat, out,
        N, IC, H, W,
        OC, KH, KW,
        OH, OW,
        OUT_HW, IC_KH_KW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.cuda()
        return conv2d_relu_bias(x, self.conv.weight, self.conv.bias, self.bias)