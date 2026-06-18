import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SPATIAL': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SPATIAL': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SPATIAL': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SPATIAL': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SPATIAL': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SPATIAL': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SPATIAL': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SPATIAL': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SPATIAL': 2048}, num_warps=8, num_stages=2),
    ],
    key=['N', 'IC', 'IH', 'IW', 'OC'],
)
@triton.jit
def conv_gelu_avgpool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    inv,
    N, IC, IH, IW,
    OC,
    OH: tl.constexpr, OW: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    KSIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
):
    # grid: (N, OC, NUM_SPLITS)
    n = tl.program_id(0)
    oc = tl.program_id(1)
    split = tl.program_id(2)

    OHOW: tl.constexpr = OH * OW

    # Load weights for this oc
    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < KSIZE
    w_base = oc * IC_C * KH * KW
    w_vals = tl.load(w_ptr + w_base + k_offs, mask=k_mask, other=0.0)

    bias = tl.load(b_ptr + oc)

    # decode k_offs into (ic, kh, kw)
    ic_idx = k_offs // (KH * KW)
    rem = k_offs % (KH * KW)
    kh_idx = rem // KW
    kw_idx = rem % KW

    # split spatial range across NUM_SPLITS programs
    per_split = (OHOW + NUM_SPLITS - 1) // NUM_SPLITS
    sp_start = split * per_split
    sp_end = tl.minimum(sp_start + per_split, OHOW)

    acc = tl.zeros((), dtype=tl.float32)

    sp_lo = sp_start
    while sp_lo < sp_end:
        sp_offs = sp_lo + tl.arange(0, BLOCK_SPATIAL)
        sp_mask = sp_offs < sp_end
        oh = sp_offs // OW
        ow = sp_offs % OW

        ih = oh[:, None] + kh_idx[None, :]
        iw = ow[:, None] + kw_idx[None, :]
        ic_b = ic_idx[None, :].broadcast_to(BLOCK_SPATIAL, BLOCK_K)

        x_idx = ((n * IC + ic_b) * IH + ih) * IW + iw
        x_mask = sp_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)

        prod = x_vals * w_vals[None, :]
        conv_out = tl.sum(prod, axis=1) + bias

        # GELU tanh approx
        c0 = 0.7978845608028654
        c1 = 0.044715
        x3 = conv_out * conv_out * conv_out
        inner = c0 * (conv_out + c1 * x3)
        t = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
        gelu = 0.5 * conv_out * (1.0 + t)

        gelu = tl.where(sp_mask, gelu, 0.0)
        acc += tl.sum(gelu, axis=0)

        sp_lo += BLOCK_SPATIAL

    result = acc * inv

    if NUM_SPLITS == 1:
        tl.store(out_ptr + n * OC + oc, result)
    else:
        tl.atomic_add(out_ptr + n * OC + oc, result)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1

        KSIZE = IC * KH * KW
        BLOCK_K = 1
        while BLOCK_K < KSIZE:
            BLOCK_K *= 2

        inv = 1.0 / float(OH * OW)

        # Choose splits to expose more parallelism. N*OC = 128*64 = 8192 is enough,
        # but more programs help latency hiding.
        OHOW = OH * OW
        # target ~16384 total programs
        target_progs = 16384
        base = N * OC
        num_splits = max(1, target_progs // base)
        # cap so each split has reasonable work
        max_splits = max(1, OHOW // 4096)
        if num_splits > max_splits:
            num_splits = max_splits
        if num_splits < 1:
            num_splits = 1

        if num_splits == 1:
            out = torch.empty((N, OC), device=x.device, dtype=x.dtype)
        else:
            out = torch.zeros((N, OC), device=x.device, dtype=x.dtype)

        grid = (N, OC, num_splits)
        conv_gelu_avgpool_kernel[grid](
            x, w, b, out,
            inv,
            N, IC, IH, IW,
            OC,
            OH, OW,
            KH, KW,
            IC,
            KSIZE,
            BLOCK_K,
            num_splits,
        )
        return out