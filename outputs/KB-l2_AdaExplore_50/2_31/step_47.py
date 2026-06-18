import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OHW', 'IC_KH_KW'],
)
@triton.jit
def _conv_fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH, KW,
    OHW, IC_KH_KW,
    constant_value, scaling_factor,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)              # batch
    pid_oc = tl.program_id(1)             # OC tile
    pid_hw = tl.program_id(2)             # OHW tile

    offs_oc = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_hw = pid_hw * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]

    oh = offs_hw // OW
    ow = offs_hw % OW

    mask_m = offs_hw < OHW
    mask_n = offs_oc < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # GEMM-K iterates over IC*KH*KW
    for k_start in range(0, IC_KH_KW, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)         # [BLOCK_K]
        mask_k = offs_k < IC_KH_KW

        # decompose k -> (ic, kh, kw)
        ic = offs_k // (KH * KW)
        rem = offs_k % (KH * KW)
        kh = rem // KW
        kw = rem % KW

        # input gather: x[n, ic, oh+kh, ow+kw]  (stride==1, padding==0)
        ih = oh[:, None] + kh[None, :]   # [BLOCK_M, BLOCK_K]
        iw = ow[:, None] + kw[None, :]   # [BLOCK_M, BLOCK_K]

        x_off = (pid_n * stride_xn
                 + ic[None, :] * stride_xc
                 + ih * stride_xh
                 + iw * stride_xw)
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [M, K]

        # weight: w[oc, ic, kh, kw]
        w_off = (offs_oc[:, None] * stride_wo
                 + ic[None, :] * stride_wi
                 + kh[None, :] * stride_wkh
                 + kw[None, :] * stride_wkw)
        w_mask = mask_n[:, None] & mask_k[None, :]
        w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # [N, K]

        acc += tl.dot(x_tile, tl.trans(w_tile))

    # bias
    b = tl.load(b_ptr + offs_oc, mask=mask_n, other=0.0)  # [BLOCK_N]
    acc = acc + b[None, :]
    # min(., constant_value) * scaling_factor  (note: original is min then +bias then *scale,
    # but here the conv-bias is the original conv bias; the extra bias is added after min)
    # We need to follow: y = min(conv_out, c) + extra_bias; here acc = conv_out so:
    acc = tl.minimum(acc, constant_value)
    # extra_bias was passed via b_ptr already? No - we used conv bias. Need separate path.
    # We'll re-add extra bias outside? Actually we pass combined bias = conv.bias added later.
    # See host: b_ptr is conv.bias only, and extra bias added in second epilogue OR fold here.
    acc = acc * scaling_factor

    # store
    out_off = (pid_n * stride_on
               + offs_oc[:, None] * stride_oc
               + oh[None, :] * stride_oh
               + ow[None, :] * stride_ow)
    # wait - shape mismatch: acc is [M, N] but out_off above is [N, M]. Fix:
    out_off = (pid_n * stride_on
               + offs_oc[None, :] * stride_oc
               + oh[:, None] * stride_oh
               + ow[:, None] * stride_ow)
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


@triton.jit
def _add_bias_kernel(
    io_ptr, bias_ptr,
    N, C, HW, total,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    c_idx = (offs // HW) % C
    x = tl.load(io_ptr + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    tl.store(io_ptr + offs, x + b, mask=mask)


def _conv_fused(x, w, conv_bias, extra_bias, constant_value, scaling_factor):
    x = x.contiguous()
    w = w.contiguous()
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = w.shape
    OH = IH - KH + 1
    OW = IW - KW + 1
    OHW = OH * OW
    IC_KH_KW = IC * KH * KW

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    cb = conv_bias.contiguous().view(-1)

    grid = lambda meta: (
        N,
        triton.cdiv(OC, meta['BLOCK_N']),
        triton.cdiv(OHW, meta['BLOCK_M']),
    )

    _conv_fused_kernel[grid](
        x, w, cb, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        OHW, IC_KH_KW,
        float(constant_value), float(scaling_factor),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    )

    # Add extra bias (after min, before *scale in reference). But reference is:
    # y = (min(conv, c) + extra_bias) * scale
    # Our kernel produced: (min(conv+conv_bias, c)) * scale = min(conv_full, c) * scale
    # We need to add extra_bias * scale.
    eb_flat = (extra_bias.contiguous().view(-1) * float(scaling_factor)).contiguous()
    HW = OH * OW
    total = N * OC * HW
    BLOCK = 1024
    grid2 = ((total + BLOCK - 1) // BLOCK,)
    _add_bias_kernel[grid2](out, eb_flat, N, OC, HW, total, BLOCK=BLOCK, num_warps=4)

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = x.contiguous().cuda()
        return _conv_fused(
            x, self.conv.weight, self.conv.bias,
            self.bias, self.constant_value, self.scaling_factor,
        )