import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 256}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OH', 'OW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv2d_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    K: tl.constexpr,  # IC*KH*KW
    K_PAD: tl.constexpr,  # padded K (power of 2)
    SUB: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_sp = tl.program_id(1)
    pid_oc = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oh = sp_offs // OW
    ow = sp_offs % OW

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (OH * OW)

    # Single K tile covering all of K (padded to power of 2)
    k_idx = tl.arange(0, K_PAD)
    k_mask = k_idx < K

    # decompose k = ic*KH*KW + kh*KW + kw
    ic = k_idx // (KH * KW)
    rem = k_idx % (KH * KW)
    kh = rem // KW
    kw = rem % KW

    # load weight tile [BLOCK_OC, K_PAD]
    w_off = oc_offs[:, None] * K + k_idx[None, :]
    w_mask = oc_mask[:, None] & k_mask[None, :]
    w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0, eviction_policy="evict_last")

    # input addresses [K_PAD, BLOCK_SP]
    ih = oh[None, :] + kh[:, None]
    iw = ow[None, :] + kw[:, None]
    x_off = (pid_n * (IC * IH * IW)
             + ic[:, None] * (IH * IW)
             + ih * IW
             + iw)
    x_mask = k_mask[:, None] & sp_mask[None, :]
    x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0, eviction_policy="evict_first")

    acc = tl.dot(w_tile, x_tile, allow_tf32=True)

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


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


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
        K = IC * KH * KW
        K_PAD = max(16, _next_pow2(K))

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        sub = float(self.subtract_value_1 + self.subtract_value_2)

        grid = lambda meta: (
            N,
            triton.cdiv(OH * OW, meta['BLOCK_SP']),
            triton.cdiv(OC, meta['BLOCK_OC']),
        )

        conv2d_mish_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW, K, K_PAD,
            sub,
        )
        return out