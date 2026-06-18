import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 64, 'GROUP_N': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64, 'GROUP_N': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64, 'GROUP_N': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64, 'GROUP_N': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64, 'GROUP_N': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64, 'GROUP_N': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64, 'GROUP_N': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_OC': 64, 'GROUP_N': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_OC': 64, 'GROUP_N': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 128, 'GROUP_N': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 128, 'GROUP_N': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 128, 'GROUP_N': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 128, 'GROUP_N': 8}, num_warps=8, num_stages=2),
    ],
    key=['N_OUT', 'OC', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv_hswish_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, IH, IW,
    OC, OH, OW,
    IC: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    N_OUT,  # B * OH * OW
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wh, stride_ww,
    stride_ob, stride_oc, stride_oh, stride_ow,
    BLOCK_N: tl.constexpr, BLOCK_OC: tl.constexpr, GROUP_N: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_oc = tl.program_id(1)

    # Swizzle pid_n for better L2 reuse of W tiles across adjacent spatial outputs
    num_pid_n = tl.cdiv(N_OUT, BLOCK_N)
    # group_id selects a contiguous block of GROUP_N programs
    num_groups = tl.cdiv(num_pid_n, GROUP_N)
    group_id = pid % num_groups
    in_group = pid // num_groups
    pid_n = group_id * GROUP_N + in_group
    pid_n = tl.where(pid_n < num_pid_n, pid_n, num_pid_n - 1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_ic = tl.arange(0, BLOCK_K)

    mask_n = offs_n < N_OUT
    mask_oc = offs_oc < OC
    mask_ic = offs_ic < IC

    # decode N axis -> (b, oh, ow)
    ow = offs_n % OW
    tmp = offs_n // OW
    oh = tmp % OH
    b = tmp // OH

    # Hoisted bases
    x_base_n = b * stride_xb + oh * stride_xh + ow * stride_xw  # [BLOCK_N]
    x_ic_off = offs_ic * stride_xc  # [BLOCK_K]
    w_ic_off = offs_ic * stride_wi  # [BLOCK_K]
    w_oc_off = offs_oc * stride_wo  # [BLOCK_OC]

    acc = tl.zeros((BLOCK_N, BLOCK_OC), dtype=tl.float32)

    # Outer loop over kh, kw — dense inner dot of K=IC (padded to BLOCK_K)
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            # X tile [BLOCK_N, BLOCK_K]
            x_off = x_base_n[:, None] + x_ic_off[None, :] + (kh * stride_xh + kw * stride_xw)
            x_mask = mask_n[:, None] & mask_ic[None, :]
            x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

            # W tile [BLOCK_K, BLOCK_OC]
            w_off = w_ic_off[:, None] + w_oc_off[None, :] + (kh * stride_wh + kw * stride_ww)
            w_mask = mask_ic[:, None] & mask_oc[None, :]
            w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

            acc += tl.dot(x_tile, w_tile, out_dtype=tl.float32)

    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + bias[None, :]

    # hardswish then relu: y = relu(x * relu6(x+3)/6)
    # since relu(hardswish(x)) = hardswish(x) for x>=0, and 0 otherwise (hardswish(x)<=0 when x<=0)
    # actually hardswish(x) = 0 when x<=-3, negative for -3<x<0, positive for x>0
    # relu(hardswish(x)) = x*(x+3)/6 clamped: for x>=3 -> x; for 0<=x<3 -> x*(x+3)/6; else 0
    x_in = acc
    relu6 = tl.minimum(tl.maximum(x_in + 3.0, 0.0), 6.0)
    hs = x_in * relu6 * (1.0 / 6.0)
    out_val = tl.maximum(hs, 0.0)

    out_off = (b * stride_ob + oh * stride_oh + ow * stride_ow)[:, None] + offs_oc[None, :] * stride_oc
    mask = mask_n[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_off, out_val, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.cuda().contiguous()
        w = self.conv.weight.cuda().contiguous()
        b = self.conv.bias.cuda().contiguous()

        B, IC, IH, IW = x.shape
        OC, _, KH, KW = w.shape
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((B, OC, OH, OW), device=x.device, dtype=x.dtype)

        N_OUT = B * OH * OW

        grid = lambda meta: (
            triton.cdiv(N_OUT, meta['BLOCK_N']),
            triton.cdiv(OC, meta['BLOCK_OC']),
        )

        # BLOCK_K = next pow2 >= IC, min 16 (tl.dot K-dim requirement)
        BLOCK_K = 1
        while BLOCK_K < IC:
            BLOCK_K *= 2
        if BLOCK_K < 16:
            BLOCK_K = 16

        conv_hswish_relu_kernel[grid](
            x, w, b, out,
            B, IH, IW,
            OC, OH, OW,
            IC, KH, KW,
            BLOCK_K,
            N_OUT,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w.stride(0), w.stride(1), w.stride(2), w.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        )
        return out