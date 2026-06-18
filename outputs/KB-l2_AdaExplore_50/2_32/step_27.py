import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=3),
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
    pid_n = tl.program_id(0)
    pid_sp = tl.program_id(1)

    offs_sp = pid_sp * BLOCK_N + tl.arange(0, BLOCK_N)
    sp_mask = offs_sp < OUT_HW
    oh = offs_sp // OW
    ow = offs_sp % OW

    x_batch_ptr = x_ptr + pid_n * stride_xn
    sp_base = oh * stride_xh + ow * stride_xw

    KHKW = KH * KW

    offs_oc = tl.arange(0, BLOCK_M)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, IC_KHKW, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < IC_KHKW

        ic = offs_k // KHKW
        khw = offs_k % KHKW
        kh = khw // KW
        kw = khw % KW

        w_ptrs = (w_ptr
                  + offs_oc[:, None] * stride_wo
                  + ic[None, :] * stride_wi
                  + kh[None, :] * stride_wkh
                  + kw[None, :] * stride_wkw)
        w_vals = tl.load(w_ptrs, mask=k_mask[None, :], other=0.0)

        k_offset = ic * stride_xc + kh * stride_xh + kw * stride_xw
        x_ptrs = x_batch_ptr + k_offset[:, None] + sp_base[None, :]
        x_load_mask = k_mask[:, None] & sp_mask[None, :]
        x_vals = tl.load(x_ptrs, mask=x_load_mask, other=0.0)

        acc += tl.dot(w_vals, x_vals)

    b_vals = tl.load(b_ptr + offs_oc)
    acc = acc + b_vals[:, None]
    acc = acc * scale

    min_val = tl.min(acc, axis=0)

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
        BLOCK_M=OC,
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