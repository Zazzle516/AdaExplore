import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'IC'],
)
@triton.jit
def conv_transpose_fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
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
):
    # M = N_batch * H_out * W_out
    # N = OC
    # Output tile: BLOCK_M output spatial positions × BLOCK_N output channels
    # Reduction: only over IC (inner), with the (kh, kw) for each output position
    # determined by (ho % STRIDE, wo % STRIDE) — exactly one (kh,kw) per output for stride==kernel,
    # but in general we iterate the valid (kh, kw) pairs.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    wo = offs_m % W_out
    tmp = offs_m // W_out
    ho = tmp % H_out
    n_idx = tmp // H_out

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # For ConvTranspose2d (no padding):
    # hi = (ho - kh) / stride, valid iff (ho - kh) % stride == 0 and 0 <= hi < H
    # We loop over (kh, kw) and for each one compute hi, wi.
    # For each output (ho, wo), there's exactly one kh in [0, KH) such that
    # (ho - kh) % STRIDE == 0 within each stride-equivalence class. So most (kh, kw)
    # pairs are invalid for a given output. But within a tile of BLOCK_M outputs,
    # different outputs hit different (kh,kw)'s.
    #
    # We iterate all KH*KW pairs but only do the GEMM-K loop over IC.

    for kh in tl.static_range(0, KH):
        ho_kh = ho - kh
        hi = ho_kh // STRIDE
        valid_h = (ho_kh >= 0) & ((ho_kh % STRIDE) == 0) & (hi < H) & (hi >= 0)
        for kw in tl.static_range(0, KW):
            wo_kw = wo - kw
            wi = wo_kw // STRIDE
            valid_w = (wo_kw >= 0) & ((wo_kw % STRIDE) == 0) & (wi < W) & (wi >= 0)
            valid = valid_h & valid_w & mask_m  # [BLOCK_M]

            # Now accumulate over IC for this (kh, kw)
            # x[n_idx, ic, hi, wi] : [BLOCK_M, IC]
            # w[ic, offs_n, kh, kw] : [IC, BLOCK_N]
            # Use a BLOCK_K-sized chunked loop over IC
            BLOCK_K: tl.constexpr = 32
            for k_start in range(0, IC, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)
                mask_k = offs_k < IC

                # x offsets: n_idx * IC*H*W + ic * H*W + hi * W + wi
                x_off = (n_idx[:, None] * (IC * H * W)
                         + offs_k[None, :] * (H * W)
                         + hi[:, None] * W
                         + wi[:, None])
                x_mask = valid[:, None] & mask_k[None, :]
                x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                # w offsets: ic * OC*KH*KW + oc * KH*KW + kh * KW + kw
                w_off = (offs_k[:, None] * (OC * KH * KW)
                         + offs_n[None, :] * (KH * KW)
                         + kh * KW + kw)
                w_mask = mask_k[:, None] & mask_n[None, :]
                w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals)

    # bias
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # epilogue
    acc = acc + add_value
    acc = tl.minimum(acc, 0.0)
    inv_sqrt2 = 0.7071067811865475
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))
    acc = acc * multiply_value

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