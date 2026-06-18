import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OUT_HW', 'IC_KHKW'],
)
@triton.jit
def conv_scale_min_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W, OC, KH, KW, OH, OW,
    scale,
    OUT_HW, IC_KHKW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program_id(0) -> batch
    # program_id(1) -> output spatial tile (BLOCK_N output positions)
    # we iterate over OC tiles inside the kernel to compute the min across all channels
    pid_n = tl.program_id(0)
    pid_sp = tl.program_id(1)

    # output spatial offsets within this tile
    offs_sp = pid_sp * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    sp_mask = offs_sp < OUT_HW
    oh = offs_sp // OW
    ow = offs_sp % OW
    # base x offset per spatial position (without ic, kh, kw contributions)
    x_sp_base = pid_n * stride_xn + oh * stride_xh + ow * stride_xw  # [BLOCK_N]

    # We will compute, for each output spatial position, the min over OC of (conv + bias) * scale
    # min_val[BLOCK_N], initialized to +inf
    min_val = tl.full([BLOCK_N], float('inf'), dtype=tl.float32)

    # iterate OC in tiles of BLOCK_M
    for oc_start in range(0, OC, BLOCK_M):
        offs_oc = oc_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
        oc_mask = offs_oc < OC

        # Accumulator [BLOCK_M, BLOCK_N]
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        # iterate over IC*KH*KW in BLOCK_K chunks
        for k_start in range(0, IC_KHKW, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
            k_mask = offs_k < IC_KHKW

            ic = offs_k // (KH * KW)
            khw = offs_k % (KH * KW)
            kh = khw // KW
            kw = khw % KW

            # weight: [BLOCK_M, BLOCK_K], w[oc, ic, kh, kw]
            w_ptrs = (w_ptr
                      + offs_oc[:, None] * stride_wo
                      + ic[None, :] * stride_wi
                      + kh[None, :] * stride_wkh
                      + kw[None, :] * stride_wkw)
            w_load_mask = oc_mask[:, None] & k_mask[None, :]
            w_vals = tl.load(w_ptrs, mask=w_load_mask, other=0.0)

            # input: [BLOCK_K, BLOCK_N], x[n, ic, oh+kh, ow+kw]
            k_offset = ic * stride_xc + kh * stride_xh + kw * stride_xw  # [BLOCK_K]
            x_ptrs = x_ptr + x_sp_base[None, :] + k_offset[:, None]
            x_load_mask = k_mask[:, None] & sp_mask[None, :]
            x_vals = tl.load(x_ptrs, mask=x_load_mask, other=0.0)

            acc += tl.dot(w_vals, x_vals)

        # add bias
        b_vals = tl.load(b_ptr + offs_oc, mask=oc_mask, other=0.0)
        acc = acc + b_vals[:, None]
        acc = acc * scale

        # mask out invalid oc rows with +inf so they don't affect min
        acc = tl.where(oc_mask[:, None], acc, float('inf'))

        # reduce along OC dim (axis=0) -> [BLOCK_N]
        tile_min = tl.min(acc, axis=0)
        min_val = tl.minimum(min_val, tile_min)

    # store output [N, 1, OH, OW]
    out_ptrs = out_ptr + pid_n * OUT_HW + offs_sp
    tl.store(out_ptrs, min_val, mask=sp_mask)


def conv_scale_min(x, weight, bias, scale):
    N, IC, H, W = x.shape
    OC, _, KH, KW = weight.shape
    OH = H - KH + 1
    OW = W - KW + 1
    OUT_HW = OH * OW
    IC_KHKW = IC * KH * KW

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    out = torch.empty((N, 1, OH, OW), device=x.device, dtype=torch.float32)

    grid = lambda meta: (N, triton.cdiv(OUT_HW, meta['BLOCK_N']))

    conv_scale_min_kernel[grid](
        x, weight, bias, out,
        N, IC, H, W, OC, KH, KW, OH, OW,
        scale,
        OUT_HW, IC_KHKW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()
        return conv_scale_min(x, w, b, float(self.scale_factor))