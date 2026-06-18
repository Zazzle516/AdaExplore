import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['N', 'OC', 'IC', 'OH', 'OW'],
)
@triton.jit
def _conv3x3_mish_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wh, stride_ww,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr,  # OC tile
    BLOCK_N: tl.constexpr,  # spatial tile
    BLOCK_K: tl.constexpr,  # IC tile
):
    pid_n = tl.program_id(0)        # batch
    pid_oc = tl.program_id(1)       # OC tile id
    pid_sp = tl.program_id(2)       # spatial tile id

    offs_oc = pid_oc * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_sp = pid_sp * BLOCK_N + tl.arange(0, BLOCK_N)

    oh = offs_sp // OW
    ow = offs_sp - oh * OW
    sp_mask = offs_sp < (OH * OW)
    oc_mask = offs_oc < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x_base = x_ptr + pid_n * stride_xn
    out_base = out_ptr + pid_n * stride_on

    # Loop over IC blocks; kh,kw unrolled
    for ic_start in range(0, IC, BLOCK_K):
        offs_ic = ic_start + tl.arange(0, BLOCK_K)
        ic_mask = offs_ic < IC

        for kh in tl.static_range(0, 3):
            ih = oh + kh
            for kw in tl.static_range(0, 3):
                iw = ow + kw
                # x: [BLOCK_K, BLOCK_N]
                x_ptrs = (x_base
                          + offs_ic[:, None] * stride_xc
                          + ih[None, :] * stride_xh
                          + iw[None, :] * stride_xw)
                x_mask = ic_mask[:, None] & sp_mask[None, :]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

                # w: [BLOCK_M, BLOCK_K]
                w_ptrs = (w_ptr
                          + offs_oc[:, None] * stride_wo
                          + offs_ic[None, :] * stride_wi
                          + kh * stride_wh
                          + kw * stride_ww)
                w_mask = oc_mask[:, None] & ic_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

                acc += tl.dot(w_vals, x_vals, allow_tf32=True)

    # bias
    b = tl.load(b_ptr + offs_oc, mask=oc_mask, other=0.0)
    acc = acc + b[:, None]

    # double mish: mish(x) = x * tanh(softplus(x)); stable softplus
    sp1 = tl.where(acc > 20.0, acc, tl.log(1.0 + tl.exp(acc)))
    t1 = 2.0 * tl.sigmoid(2.0 * sp1) - 1.0
    y = acc * t1
    sp2 = tl.where(y > 20.0, y, tl.log(1.0 + tl.exp(y)))
    t2 = 2.0 * tl.sigmoid(2.0 * sp2) - 1.0
    z = y * t2

    out_ptrs = (out_base
                + offs_oc[:, None] * stride_oc
                + oh[None, :] * stride_oh
                + ow[None, :] * stride_ow)
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptrs, z, mask=out_mask)


def conv3x3_mish_mish(x, weight, bias):
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = weight.shape
    OH = IH - KH + 1
    OW = IW - KW + 1

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        N,
        triton.cdiv(OC, meta['BLOCK_M']),
        triton.cdiv(OH * OW, meta['BLOCK_N']),
    )

    _conv3x3_mish_mish_kernel[grid](
        x, weight, bias, out,
        N, IC, IH, IW,
        OC, OH, OW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return conv3x3_mish_mish(x, self.conv.weight, self.conv.bias)