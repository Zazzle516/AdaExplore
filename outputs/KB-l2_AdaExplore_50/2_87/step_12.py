import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128}, num_warps=4, num_stages=3),
    ],
    key=['OC', 'OH', 'OW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv2d_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    SUB: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_sp = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_n = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oh = sp_offs // OW
    ow = sp_offs % OW

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (OH * OW)

    K = IC * KH * KW  # GEMM-K

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # K is small (IC*KH*KW = 8*3*3 = 72). Iterate K in BLOCK_K chunks.
    k_range = tl.arange(0, BLOCK_K)
    x_base = pid_n * (IC * IH * IW)

    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + k_range  # [BLOCK_K]
        k_mask = k_idx < K

        # Decompose k -> (ic, kh, kw)
        ic = k_idx // (KH * KW)
        rem = k_idx % (KH * KW)
        kh = rem // KW
        kw = rem % KW

        # Weight tile [BLOCK_OC, BLOCK_K]
        w_off = oc_offs[:, None] * K + k_idx[None, :]
        w_mask = oc_mask[:, None] & k_mask[None, :]
        w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

        # Input tile [BLOCK_K, BLOCK_SP]: for each k, load input at (n, ic, oh+kh, ow+kw)
        ih = oh[None, :] + kh[:, None]  # [BLOCK_K, BLOCK_SP]
        iw = ow[None, :] + kw[:, None]
        x_off = x_base + ic[:, None] * (IH * IW) + ih * IW + iw
        x_mask = k_mask[:, None] & sp_mask[None, :]
        x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

        acc += tl.dot(w_tile, x_tile, allow_tf32=True)

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[:, None]
    acc = acc - SUB

    # mish: x * tanh(softplus(x))
    sp_val = tl.log(1.0 + tl.exp(acc))
    e2 = tl.exp(2.0 * sp_val)
    tanh_val = (e2 - 1.0) / (e2 + 1.0)
    out = acc * tanh_val

    out_off = (pid_n * OC * OH * OW
               + oc_offs[:, None] * (OH * OW)
               + sp_offs[None, :])
    mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        sub = float(self.subtract_value_1 + self.subtract_value_2)

        # K = IC*KH*KW = 72; pad to 128 for tl.dot
        K = IC * KH * KW
        # Round BLOCK_K up to a power of 2 >= K (Triton tl.dot requires power-of-2 dims, min 16)
        BLOCK_K = 1
        while BLOCK_K < K:
            BLOCK_K *= 2
        if BLOCK_K < 16:
            BLOCK_K = 16

        grid = lambda meta: (
            triton.cdiv(OH * OW, meta['BLOCK_SP']),
            triton.cdiv(OC, meta['BLOCK_OC']),
            N,
        )

        conv2d_mish_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            sub,
            BLOCK_K=BLOCK_K,
        )
        return out