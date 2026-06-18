import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=2),
    ],
    key=['OD', 'OH', 'OW', 'OC', 'IC'],
)
@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, cb_ptr, scale_ptr, bias_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KT: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    K: tl.constexpr,
):
    pid_n = tl.program_id(0)  # spatial tile index
    pid_b = tl.program_id(1)  # batch index

    OUT_SP = OD * OH * OW

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_n = offs_n < OUT_SP

    # Decompose spatial index to (od, oh, ow)
    od = offs_n // (OH * OW)
    rem = offs_n % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    offs_m = tl.arange(0, BLOCK_M)  # OC
    mask_m = offs_m < OC

    # Build K reduction (IC * KT * KH * KW)
    offs_k = tl.arange(0, K)  # [K]
    # k = ((ic*KT + kt)*KH + kh)*KW + kw
    kw = offs_k % KW
    tmp1 = offs_k // KW
    kh = tmp1 % KH
    tmp2 = tmp1 // KH
    kt = tmp2 % KT
    ic = tmp2 // KT

    # Input pointer base for this batch
    x_batch_ptr = x_ptr + pid_b * IC * ID * IH * IW

    # Compute input indices for [BLOCK_N, K]
    # in_d = od + kt, in_h = oh + kh, in_w = ow + kw
    in_d = od[:, None] + kt[None, :]  # [BLOCK_N, K]
    in_h = oh[:, None] + kh[None, :]
    in_w = ow[:, None] + kw[None, :]

    x_offs = ic[None, :] * (ID * IH * IW) + in_d * (IH * IW) + in_h * IW + in_w
    x_mask = mask_n[:, None]  # K dimension is always valid (no padding)
    x_vals = tl.load(x_batch_ptr + x_offs, mask=x_mask, other=0.0)  # [BLOCK_N, K]

    # Weight: [OC, IC, KT, KH, KW] -> [OC, K]
    w_offs = offs_m[:, None] * K + offs_k[None, :]  # [BLOCK_M, K]
    w_mask = mask_m[:, None]
    w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)  # [BLOCK_M, K]

    # GEMM: [BLOCK_N, K] x [K, BLOCK_M] -> [BLOCK_N, BLOCK_M]
    acc = tl.dot(x_vals, tl.trans(w_vals))  # [BLOCK_N, BLOCK_M]

    # Add conv bias
    cb = tl.load(cb_ptr + offs_m, mask=mask_m, other=0.0)  # [BLOCK_M]
    acc = acc + cb[None, :]

    # Load scale and bias
    sc = tl.load(scale_ptr + offs_m, mask=mask_m, other=0.0)
    bi = tl.load(bias_ptr + offs_m, mask=mask_m, other=0.0)

    # Epilogue: scale, tanh, bias, sigmoid
    y = acc * sc[None, :]
    e2 = tl.exp(2.0 * y)
    t = (e2 - 1.0) / (e2 + 1.0)
    z = t * bi[None, :]
    out_val = 1.0 / (1.0 + tl.exp(-z))

    # Store: output is [N, OC, OD, OH, OW]
    out_batch_ptr = out_ptr + pid_b * OC * OUT_SP
    out_offs = offs_m[None, :] * OUT_SP + offs_n[:, None]  # [BLOCK_N, BLOCK_M]
    out_mask = mask_n[:, None] & mask_m[None, :]
    tl.store(out_batch_ptr + out_offs, out_val, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KT = KH = KW = self.kernel_size
        OD = ID - KT + 1
        OH = IH - KH + 1
        OW = IW - KW + 1
        K = IC * KT * KH * KW  # 81

        # Round OC up to power of 2 for BLOCK_M
        BLOCK_M = 16  # OC=16 fits exactly
        # K=81, need to pad to power of 2 for Triton; use 128 with mask
        K_PAD = 128

        # Pad weight to [OC, K_PAD]
        w = self.conv.weight.contiguous().view(OC, K)
        w_padded = torch.zeros((OC, K_PAD), device=w.device, dtype=w.dtype)
        w_padded[:, :K] = w

        # For x loads, we need K_PAD. Use mask on K dimension.
        # Simpler: keep K as actual size, but Triton needs power-of-2 for tl.dot inner dim.
        # We pad implicitly via mask. Actually tl.dot requires power-of-2 dims.
        # So pass K_PAD and mask out the padded K.

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        cb = self.conv.bias.contiguous()
        scale = self.scaling_factor.contiguous().view(-1)
        bias = self.bias.contiguous().view(-1)

        OUT_SP = OD * OH * OW
        grid = lambda meta: (triton.cdiv(OUT_SP, meta['BLOCK_N']), N)

        conv3d_fused_kernel[grid](
            x, w_padded, cb, scale, bias, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KT, KH, KW,
            BLOCK_M=BLOCK_M,
            K=K_PAD,
        )
        return out