import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_IC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_IC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_IC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_IC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_IC': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'IC'],
)
@triton.jit
def conv_transpose_fused_kernel(
    x_ptr,          # input (N, IC, H, W) contiguous
    w_ptr,          # weight (IC, OC, KH, KW) contiguous
    b_ptr,          # bias (OC,)
    out_ptr,        # output (N, OC, H_out, W_out) contiguous
    M, N,
    H, W,
    H_out, W_out,
    IC, OC,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE: tl.constexpr,
    add_value: tl.constexpr,
    multiply_value: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # decode m -> (n_idx, ho, wo)
    wo = offs_m % W_out
    tmp = offs_m // W_out
    ho = tmp % H_out
    n_idx = tmp // H_out

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Hoisted base offsets
    HW = H * W
    OC_KHKW = OC * KH * KW
    x_n_base = n_idx * (IC * HW)  # [BLOCK_M]
    w_n_base = offs_n * (KH * KW)  # [BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Unroll over (kh, kw)
    for kh_idx in tl.static_range(0, KH):
        ho_kh = ho - kh_idx
        hi = ho_kh // STRIDE
        valid_h = (ho_kh >= 0) & ((ho_kh % STRIDE) == 0) & (hi < H) & (hi >= 0)
        for kw_idx in tl.static_range(0, KW):
            wo_kw = wo - kw_idx
            wi = wo_kw // STRIDE
            valid_w = (wo_kw >= 0) & ((wo_kw % STRIDE) == 0) & (wi < W) & (wi >= 0)
            valid_m = valid_h & valid_w & mask_m  # [BLOCK_M]

            # x base offset (without ic): n_idx*IC*HW + hi*W + wi
            x_spatial = x_n_base + hi * W + wi  # [BLOCK_M]
            # w base offset (without ic): oc*KH*KW + kh*KW + kw
            w_kkw = w_n_base + (kh_idx * KW + kw_idx)  # [BLOCK_N]

            # Inner loop over IC in BLOCK_IC chunks
            for ic_start in range(0, IC, BLOCK_IC):
                offs_ic = ic_start + tl.arange(0, BLOCK_IC)
                mask_ic = offs_ic < IC

                x_off = x_spatial[:, None] + offs_ic[None, :] * HW
                x_mask = valid_m[:, None] & mask_ic[None, :]
                x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                w_off = offs_ic[:, None] * OC_KHKW + w_kkw[None, :]
                w_mask = mask_ic[:, None] & mask_n[None, :]
                w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals)

    # add bias
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # epilogue: + add_value, min(.,0), GELU, * multiply_value
    acc = acc + add_value
    acc = tl.minimum(acc, 0.0)
    inv_sqrt2 = 0.7071067811865475
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))
    acc = acc * multiply_value

    # store to out: out[n_idx, oc, ho, wo]
    out_off = (n_idx[:, None] * (OC * H_out * W_out)
               + offs_n[None, :] * (H_out * W_out)
               + ho[:, None] * W_out
               + wo[:, None])
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


def conv_transpose_fused(x, weight, bias, stride, add_value, multiply_value):
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    N_b, IC, H, W = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w

    H_out = (H - 1) * stride + KH
    W_out = (W - 1) * stride + KW

    out = torch.empty((N_b, OC, H_out, W_out), device=x.device, dtype=x.dtype)

    M = N_b * H_out * W_out
    N = OC

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    conv_transpose_fused_kernel[grid](
        x, weight, bias, out,
        M, N,
        H, W,
        H_out, W_out,
        IC, OC,
        KH, KW,
        stride,
        float(add_value), float(multiply_value),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value
        self.stride = stride

    def forward(self, x):
        return conv_transpose_fused(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.stride,
            self.add_value,
            self.multiply_value,
        )