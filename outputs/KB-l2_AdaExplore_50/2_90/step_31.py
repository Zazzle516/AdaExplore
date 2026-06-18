import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=8, num_stages=2),
    ],
    key=['total'],
)
@triton.jit
def fused_epilogue_kernel(
    x_ptr, sum_ptr, bias_ptr, out_ptr,
    total,
    SPATIAL: tl.constexpr,
    C_MASK: tl.constexpr,
    LOG2_SPATIAL: tl.constexpr,
    NEG_SLOPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    c_idx = (offs >> LOG2_SPATIAL) & C_MASK

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    s = tl.load(sum_ptr + c_idx, mask=mask, other=0.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    x = x + b
    x = tl.where(x >= 0, x, x * NEG_SLOPE)
    x = x + s
    x = tl.maximum(x, -1.0)
    x = tl.minimum(x, 1.0)
    inv_sqrt2 = 0.70710678118654752440
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(out_ptr + offs, x, mask=mask)


@triton.jit
def fused_epilogue_kernel_generic(
    x_ptr, sum_ptr, bias_ptr, out_ptr,
    total, spatial, C,
    NEG_SLOPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    c_idx = (offs // spatial) % C

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    s = tl.load(sum_ptr + c_idx, mask=mask, other=0.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    x = x + b
    x = tl.where(x >= 0, x, x * NEG_SLOPE)
    x = x + s
    x = tl.maximum(x, -1.0)
    x = tl.minimum(x, 1.0)
    inv_sqrt2 = 0.70710678118654752440
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(out_ptr + offs, x, mask=mask)


def _is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def fused_epilogue(x, sum_tensor, bias, neg_slope=0.2):
    x = x.contiguous()
    sum_flat = sum_tensor.contiguous().view(-1)
    bias_flat = bias.contiguous().view(-1)
    out = x  # in-place
    N, C, D, H, W = x.shape
    spatial = D * H * W
    total = x.numel()

    if _is_pow2(C) and _is_pow2(spatial):
        log2_spatial = int(math.log2(spatial))
        c_mask = C - 1
        grid = lambda meta: ((total + meta['BLOCK'] - 1) // meta['BLOCK'],)
        fused_epilogue_kernel[grid](
            x, sum_flat, bias_flat, out,
            total,
            SPATIAL=spatial,
            C_MASK=c_mask,
            LOG2_SPATIAL=log2_spatial,
            NEG_SLOPE=neg_slope,
        )
    else:
        BLOCK = 4096
        grid = ((total + BLOCK - 1) // BLOCK,)
        fused_epilogue_kernel_generic[grid](
            x, sum_flat, bias_flat, out,
            total, spatial, C,
            NEG_SLOPE=neg_slope,
            BLOCK=BLOCK,
            num_warps=8,
        )
    return out


# ---------------- Conv3d Implicit GEMM Kernel ----------------
# Output GEMM: M = OC, N = N_batch * D' * H' * W', K = IC * kT * kH * kW
# Input layout: NCDHW contiguous
# Weight layout: (OC, IC, kT, kH, kW) contiguous -> flattened (OC, K)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'NOUT', 'K_TOTAL'],
)
@triton.jit
def conv3d_implicit_gemm_kernel(
    x_ptr, w_ptr, bias_ptr, sum_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KT, KH, KW,
    NOUT,  # OD*OH*OW per batch
    K_TOTAL,  # IC * KT * KH * KW
    # strides for input (NCDHW)
    x_str_n, x_str_c, x_str_d, x_str_h, x_str_w,
    # strides for output
    o_str_n, o_str_c, o_str_d, o_str_h, o_str_w,
    NEG_SLOPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    KHW: tl.constexpr,
    HW_OUT: tl.constexpr,  # OH * OW
    OW_C: tl.constexpr,
):
    pid_n_batch = tl.program_id(2)  # batch index
    pid_m = tl.program_id(0)  # OC tile
    pid_n = tl.program_id(1)  # spatial tile

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # OC indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # spatial indices within batch

    # Decompose spatial index into (d, h, w)
    od = offs_n // HW_OUT
    rem = offs_n - od * HW_OUT
    oh = rem // OW_C
    ow = rem - oh * OW_C

    mask_m = offs_m < OC
    mask_n = offs_n < NOUT

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K iteration: K = IC * KT * KH * KW
    # We iterate K in BLOCK_K chunks
    for k_start in range(0, K_TOTAL, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_TOTAL

        # Decompose k into (ic, kt, kh, kw)
        ic = offs_k // KHW
        krem = offs_k - ic * KHW
        kt = krem // (KH * KW)
        krem2 = krem - kt * (KH * KW)
        kh = krem2 // KW
        kw = krem2 - kh * KW

        # Weight: (OC, K) -> w_ptr + offs_m[:, None] * K_TOTAL + offs_k[None, :]
        w_offs = offs_m[:, None] * K_TOTAL + offs_k[None, :]
        w_mask = mask_m[:, None] & mask_k[None, :]
        w_tile = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

        # Input gather: for each (n_idx, k_idx), input at batch=pid_n_batch, channel=ic, d=od+kt, h=oh+kh, w=ow+kw
        # No padding, so no bounds on input spatial except already valid due to OD = ID - KT + 1 etc.
        in_d = od[None, :] + kt[:, None]
        in_h = oh[None, :] + kh[:, None]
        in_w = ow[None, :] + kw[:, None]

        x_offs = (pid_n_batch * x_str_n
                  + ic[:, None] * x_str_c
                  + in_d * x_str_d
                  + in_h * x_str_h
                  + in_w * x_str_w)

        x_mask = mask_k[:, None] & mask_n[None, :]
        x_tile = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)  # (BLOCK_K, BLOCK_N)

        acc += tl.dot(w_tile, x_tile, allow_tf32=True)

    # Epilogue: add bias, leaky_relu, add sum_tensor, clamp, gelu
    bias = tl.load(bias_ptr + offs_m, mask=mask_m, other=0.0)
    sumt = tl.load(sum_ptr + offs_m, mask=mask_m, other=0.0)

    acc = acc + bias[:, None]
    acc = tl.where(acc >= 0, acc, acc * NEG_SLOPE)
    acc = acc + sumt[:, None]
    acc = tl.maximum(acc, -1.0)
    acc = tl.minimum(acc, 1.0)
    inv_sqrt2 = 0.70710678118654752440
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # Store: output is (N, OC, OD, OH, OW) contiguous
    # out_offs = batch * OC*NOUT + oc * NOUT + spatial
    out_offs = (pid_n_batch * o_str_n
                + offs_m[:, None] * o_str_c
                + od[None, :] * o_str_d
                + oh[None, :] * o_str_h
                + ow[None, :] * o_str_w)
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


def conv3d_fused(x, weight, bias, sum_tensor, neg_slope=0.2):
    x = x.contiguous()
    weight = weight.contiguous()
    N, IC, ID, IH, IW = x.shape
    OC, _, KT, KH, KW = weight.shape
    OD = ID - KT + 1
    OH = IH - KH + 1
    OW = IW - KW + 1
    NOUT = OD * OH * OW
    K_TOTAL = IC * KT * KH * KW
    HW_OUT = OH * OW
    KHW = KT * KH * KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)
    bias_flat = bias.contiguous().view(-1)
    sum_flat = sum_tensor.contiguous().view(-1)

    # Strides (in elements)
    x_str = x.stride()
    o_str = out.stride()

    grid = lambda meta: (
        triton.cdiv(OC, meta['BLOCK_M']),
        triton.cdiv(NOUT, meta['BLOCK_N']),
        N,
    )

    conv3d_implicit_gemm_kernel[grid](
        x, weight, bias_flat, sum_flat, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KT, KH, KW,
        NOUT, K_TOTAL,
        x_str[0], x_str[1], x_str[2], x_str[3], x_str[4],
        o_str[0], o_str[1], o_str[2], o_str[3], o_str[4],
        NEG_SLOPE=neg_slope,
        KHW=KHW,
        HW_OUT=HW_OUT,
        OW_C=OW,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super().__init__()
        conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.weight = nn.Parameter(conv.weight.detach().clone())
        self.bias = nn.Parameter(conv.bias.detach().clone())
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        return conv3d_fused(x, self.weight, self.bias, self.sum_tensor, neg_slope=0.2)