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
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=2, num_stages=4),
    ],
    key=['M', 'N', 'K_PER_PARITY'],
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
    K_PER_PARITY: tl.constexpr,   # IC * KH_PAR * KW_PAR
    KH: tl.constexpr,
    KW: tl.constexpr,
    KH_PAR: tl.constexpr,   # ceil(KH/STRIDE)
    KW_PAR: tl.constexpr,   # ceil(KW/STRIDE)
    STRIDE: tl.constexpr,
    add_value: tl.constexpr,
    multiply_value: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Specialized for STRIDE=2, KH=KW=4 → KH_PAR=KW_PAR=2
    # For each output (ho, wo), the valid (kh, kw) pairs are those with
    # kh % STRIDE == ho % STRIDE and kw % STRIDE == wo % STRIDE.
    # We split the output grid by (ho_par, wo_par) ∈ {0..STRIDE-1}^2 = 4 partitions,
    # and inside each partition only K_PER_PARITY = IC * KH_PAR * KW_PAR taps are valid.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_p = tl.program_id(2)   # parity id, in [0, STRIDE*STRIDE)

    ho_par = pid_p // STRIDE
    wo_par = pid_p % STRIDE

    # M_par = number of output (n_idx, ho, wo) with ho%STRIDE==ho_par and wo%STRIDE==wo_par
    H_par = (H_out + STRIDE - 1 - ho_par) // STRIDE  # = ceil((H_out - ho_par)/STRIDE)
    W_par = (W_out + STRIDE - 1 - wo_par) // STRIDE

    # Build offs_m within this parity
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    M_par = M // (STRIDE * STRIDE)  # but actual depends; recompute below
    # Actually M_par = N_b * H_par * W_par
    # We need N_b: N_b = M / (H_out*W_out)
    HW_out = H_out * W_out
    N_b = M // HW_out
    M_par_real = N_b * H_par * W_par

    mask_m = offs_m < M_par_real

    # decode offs_m -> (n_idx, hi_par_idx, wi_par_idx)
    wi_par_idx = offs_m % W_par
    tmp = offs_m // W_par
    hi_par_idx = tmp % H_par
    n_idx = tmp // H_par

    ho = hi_par_idx * STRIDE + ho_par
    wo = wi_par_idx * STRIDE + wo_par

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction K = IC * KH_PAR * KW_PAR
    # k indexes (ic, kh_idx, kw_idx) where kh = kh_idx*STRIDE + ho_par, kw = kw_idx*STRIDE + wo_par
    BLOCK_K: tl.constexpr = K_PER_PARITY  # entire reduction in one shot (256 for IC=64,KH_PAR=KW_PAR=2)

    offs_k = tl.arange(0, BLOCK_K)
    # decode
    kw_idx = offs_k % KW_PAR
    tmp_k = offs_k // KW_PAR
    kh_idx = tmp_k % KH_PAR
    ic = tmp_k // KH_PAR

    kh = kh_idx * STRIDE + ho_par
    kw = kw_idx * STRIDE + wo_par

    # hi = (ho - kh)/STRIDE = hi_par_idx - kh_idx
    # wi = (wo - kw)/STRIDE = wi_par_idx - kw_idx
    hi = hi_par_idx[:, None] - kh_idx[None, :]
    wi = wi_par_idx[:, None] - kw_idx[None, :]

    valid = (hi >= 0) & (hi < H) & (wi >= 0) & (wi < W) & mask_m[:, None]

    x_off = (n_idx[:, None] * (IC * H * W)
             + ic[None, :] * (H * W)
             + hi * W
             + wi)
    x_vals = tl.load(x_ptr + x_off, mask=valid, other=0.0)

    # weight offset: ic*OC*KH*KW + oc*KH*KW + kh*KW + kw
    w_off = (ic[:, None] * (OC * KH * KW)
             + offs_n[None, :] * (KH * KW)
             + kh[:, None] * KW
             + kw[:, None])
    w_mask = mask_n[None, :]
    w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

    acc = tl.dot(x_vals, w_vals, acc)

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

    KH_PAR = (KH + stride - 1) // stride
    KW_PAR = (KW + stride - 1) // stride
    K_PER_PARITY = IC * KH_PAR * KW_PAR

    # M is total output spatial-batch; we'll split by parity inside the kernel
    M = N_b * H_out * W_out
    N = OC

    # M_par_max upper bound
    H_par_max = (H_out + stride - 1) // stride
    W_par_max = (W_out + stride - 1) // stride
    M_par_max = N_b * H_par_max * W_par_max

    grid = lambda meta: (
        triton.cdiv(M_par_max, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
        stride * stride,
    )

    conv_transpose_fused_kernel[grid](
        x, weight, bias, out,
        M, N,
        H, W,
        H_out, W_out,
        IC, OC,
        K_PER_PARITY,
        KH, KW,
        KH_PAR, KW_PAR,
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