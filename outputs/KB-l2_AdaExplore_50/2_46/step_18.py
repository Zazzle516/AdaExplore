import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'OC', 'POH', 'POW'],
)
@triton.jit
def fused_conv_tanh_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    POH, POW,
    KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    SUB1: tl.constexpr, SUB2: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    k_offs = tl.arange(0, BLOCK_K)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (POH * POW)

    sp_safe = tl.where(sp_mask, sp_offs, 0)
    poh = sp_safe // POW
    pow_ = sp_safe % POW

    K_TOTAL = IC * KH * KW

    acc00 = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    x_batch_off = pid_n * (IC * IH * IW)

    for k_start in range(0, K_TOTAL, BLOCK_K):
        k_idx = k_start + k_offs  # [BLOCK_K]
        k_mask = k_idx < K_TOTAL

        ic = k_idx // (KH * KW)
        kh_kw = k_idx % (KH * KW)
        kh = kh_kw // KW
        kw = kh_kw % KW

        # weight tile [BLOCK_OC, BLOCK_K]
        w_off = oc_offs[:, None] * (IC * KH * KW) + k_idx[None, :]
        w_tile = tl.load(w_ptr + w_off, mask=oc_mask[:, None] & k_mask[None, :], other=0.0)

        # base ih, iw rows (depend on k)
        ic_off = ic * (IH * IW)  # [BLOCK_K]
        kh_row = kh  # [BLOCK_K]
        kw_row = kw  # [BLOCK_K]

        # pool sub (0,0)
        ph0 = poh * POOL
        pw0 = pow_ * POOL
        ih = ph0[None, :] + kh_row[:, None]
        iw = pw0[None, :] + kw_row[:, None]
        x_off = x_batch_off + ic_off[:, None] + ih * IW + iw
        x_tile = tl.load(x_ptr + x_off, mask=k_mask[:, None] & sp_mask[None, :], other=0.0)
        acc00 = tl.dot(w_tile, x_tile, acc00)

        # pool sub (0,1)
        iw1 = (pw0 + 1)[None, :] + kw_row[:, None]
        x_off = x_batch_off + ic_off[:, None] + ih * IW + iw1
        x_tile = tl.load(x_ptr + x_off, mask=k_mask[:, None] & sp_mask[None, :], other=0.0)
        acc01 = tl.dot(w_tile, x_tile, acc01)

        # pool sub (1,0)
        ih1 = (ph0 + 1)[None, :] + kh_row[:, None]
        x_off = x_batch_off + ic_off[:, None] + ih1 * IW + iw
        x_tile = tl.load(x_ptr + x_off, mask=k_mask[:, None] & sp_mask[None, :], other=0.0)
        acc10 = tl.dot(w_tile, x_tile, acc10)

        # pool sub (1,1)
        x_off = x_batch_off + ic_off[:, None] + ih1 * IW + iw1
        x_tile = tl.load(x_ptr + x_off, mask=k_mask[:, None] & sp_mask[None, :], other=0.0)
        acc11 = tl.dot(w_tile, x_tile, acc11)

    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)[:, None]

    # apply bias, sub1, tanh, sub2 to each accumulator
    v00 = acc00 + b_val - SUB1
    e = tl.exp(2.0 * v00)
    v00 = (e - 1.0) / (e + 1.0) - SUB2

    v01 = acc01 + b_val - SUB1
    e = tl.exp(2.0 * v01)
    v01 = (e - 1.0) / (e + 1.0) - SUB2

    v10 = acc10 + b_val - SUB1
    e = tl.exp(2.0 * v10)
    v10 = (e - 1.0) / (e + 1.0) - SUB2

    v11 = acc11 + b_val - SUB1
    e = tl.exp(2.0 * v11)
    v11 = (e - 1.0) / (e + 1.0) - SUB2

    out_val = (v00 + v01 + v10 + v11) * (1.0 / (POOL * POOL))

    out_off = pid_n * (OC * POH * POW) + oc_offs[:, None] * (POH * POW) + sp_offs[None, :]
    mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, out_val, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = float(subtract1_value)
        self.subtract2_value = float(subtract2_value)
        self.kernel_size_pool = int(kernel_size_pool)
        self.kernel_size = int(kernel_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

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
        POOL = self.kernel_size_pool
        POH = OH // POOL
        POW = OW // POOL

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=torch.float32)

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_OC']),
            triton.cdiv(POH * POW, meta['BLOCK_SP']),
        )

        fused_conv_tanh_pool_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            POH, POW,
            KH, KW,
            POOL,
            self.subtract1_value, self.subtract2_value,
        )
        return out