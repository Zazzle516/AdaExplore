import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_transpose_fused_kernel(
    x_ptr,          # input (N, IC, H, W) contiguous
    w_ptr,          # weight (IC, OC, KH, KW) contiguous
    b_ptr,          # bias (OC,)
    out_ptr,        # output (N, OC, H_out, W_out) contiguous
    M, N, K,
    H, W,
    H_out, W_out,
    IC, OC,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE: tl.constexpr,
    add_value, multiply_value,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # M = N_batch * H_out * W_out, output spatial-batch dimension
    # N = OC, output channel dimension
    # K = IC * KH * KW, reduction dimension
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # iterate over K = IC * KH * KW in BLOCK_K chunks
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # decode k -> (ic, kh, kw)
        kw = offs_k % KW
        tmp_k = offs_k // KW
        kh = tmp_k % KH
        ic = tmp_k // KH

        # For ConvTranspose2d (no padding):
        # out[n, oc, ho, wo] = sum_{ic,kh,kw} x[n, ic, hi, wi] * w[ic, oc, kh, kw]
        # where ho = hi*stride + kh,  wo = wi*stride + kw
        # so hi = (ho - kh) / stride, valid if (ho - kh) >= 0 and divisible

        # shape [BLOCK_M, BLOCK_K]
        ho_kh = ho[:, None] - kh[None, :]
        wo_kw = wo[:, None] - kw[None, :]
        hi = ho_kh // STRIDE
        wi = wo_kw // STRIDE
        valid_h = (ho_kh >= 0) & ((ho_kh % STRIDE) == 0) & (hi < H)
        valid_w = (wo_kw >= 0) & ((wo_kw % STRIDE) == 0) & (wi < W)
        valid = valid_h & valid_w & mask_m[:, None] & mask_k[None, :]

        # x offsets: n_idx[:, None] * IC * H * W + ic[None, :] * H * W + hi * W + wi
        x_off = n_idx[:, None] * (IC * H * W) + ic[None, :] * (H * W) + hi * W + wi
        x_vals = tl.load(x_ptr + x_off, mask=valid, other=0.0)

        # weight: w[ic, oc, kh, kw] shape (IC, OC, KH, KW)
        # for each k (ic,kh,kw) and each n (oc): offset = ic*OC*KH*KW + oc*KH*KW + kh*KW + kw
        w_off = (ic[:, None] * (OC * KH * KW)
                 + offs_n[None, :] * (KH * KW)
                 + kh[:, None] * KW
                 + kw[:, None])
        w_mask = mask_k[:, None] & mask_n[None, :]
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
    K = IC * KH * KW

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    conv_transpose_fused_kernel[grid](
        x, weight, bias, out,
        M, N, K,
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