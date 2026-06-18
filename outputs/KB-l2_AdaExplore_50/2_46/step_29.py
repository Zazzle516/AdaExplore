import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 16, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 16, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 16, 'BLOCK_IC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 32, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 16, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 32, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'IC', 'POH', 'POW'],
)
@triton.jit
def fused_conv_tanh_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC,
    IH: tl.constexpr, IW: tl.constexpr,
    OC,
    OH: tl.constexpr, OW: tl.constexpr,
    POH: tl.constexpr, POW: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    SUB1: tl.constexpr, SUB2: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]
    
    # pooled output coordinates
    poh = sp_offs // POW
    pow_ = sp_offs % POW

    # Inside each pool window: POOL*POOL conv outputs
    # Real spatial tile = BLOCK_SP * POOL * POOL conv outputs
    # We expand: for each sp lane, generate POOL*POOL conv positions
    
    pool_idx = tl.arange(0, POOL * POOL)  # [POOL*POOL]
    ph = pool_idx // POOL
    pw = pool_idx % POOL
    
    # oh, ow for every (sp, pool_pos) pair: shape [BLOCK_SP, POOL*POOL]
    oh = poh[:, None] * POOL + ph[None, :]  # [BLOCK_SP, POOL*POOL]
    ow = pow_[:, None] * POOL + pw[None, :]  # [BLOCK_SP, POOL*POOL]
    
    # Flatten to [BLOCK_SP * POOL*POOL]
    SP_FULL: tl.constexpr = BLOCK_SP * POOL * POOL
    oh_flat = tl.reshape(oh, (SP_FULL,))
    ow_flat = tl.reshape(ow, (SP_FULL,))
    
    sp_mask_pooled = sp_offs < (POH * POW)  # [BLOCK_SP]
    sp_mask_full = tl.reshape(tl.broadcast_to(sp_mask_pooled[:, None], (BLOCK_SP, POOL * POOL)), (SP_FULL,))

    oc_mask = oc_offs < OC

    # Accumulator for conv output: [BLOCK_OC, SP_FULL]
    acc = tl.zeros((BLOCK_OC, SP_FULL), dtype=tl.float32)

    # Hoist base pointer for batch
    x_base = pid_n * (IC * IH * IW)
    
    # Loop over IC in blocks, plus KH, KW
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih_flat = oh_flat + kh  # [SP_FULL]
            iw_flat = ow_flat + kw  # [SP_FULL]
            # spatial offset in input: ih*IW + iw, shape [SP_FULL]
            sp_in_off = x_base + ih_flat * IW + iw_flat
            
            for ic_start in range(0, IC, BLOCK_IC):
                ic_offs = ic_start + tl.arange(0, BLOCK_IC)  # [BLOCK_IC]
                ic_mask = ic_offs < IC
                
                # Load x[n, ic, ih, iw] -> [BLOCK_IC, SP_FULL]
                x_off = ic_offs[:, None] * (IH * IW) + sp_in_off[None, :]
                x_mask = ic_mask[:, None] & sp_mask_full[None, :]
                x_val = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [BLOCK_IC, SP_FULL]
                
                # Load w[oc, ic, kh, kw] -> [BLOCK_OC, BLOCK_IC]
                w_off = oc_offs[:, None] * (IC * KH * KW) + ic_offs[None, :] * (KH * KW) + kh * KW + kw
                w_mask = oc_mask[:, None] & ic_mask[None, :]
                w_val = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # [BLOCK_OC, BLOCK_IC]
                
                # tl.dot: [BLOCK_OC, BLOCK_IC] x [BLOCK_IC, SP_FULL] -> [BLOCK_OC, SP_FULL]
                acc = tl.dot(w_val, x_val, acc)
    
    # Add bias
    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc = acc + b_val[:, None]
    
    # Apply sub1, tanh, sub2
    v = acc - SUB1
    e2 = tl.exp(2.0 * v)
    t = (e2 - 1.0) / (e2 + 1.0)
    v = t - SUB2
    
    # Reshape to [BLOCK_OC, BLOCK_SP, POOL*POOL] and sum over last dim
    v_r = tl.reshape(v, (BLOCK_OC, BLOCK_SP, POOL * POOL))
    pooled = tl.sum(v_r, axis=2) / (POOL * POOL)  # [BLOCK_OC, BLOCK_SP]
    
    # Store
    out_off = pid_n * (OC * POH * POW) + oc_offs[:, None] * (POH * POW) + sp_offs[None, :]
    mask = oc_mask[:, None] & sp_mask_pooled[None, :]
    tl.store(out_ptr + out_off, pooled, mask=mask)


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

        grid = lambda META: (N, triton.cdiv(OC, META['BLOCK_OC']), triton.cdiv(POH * POW, META['BLOCK_SP']))

        fused_conv_tanh_pool_kernel[grid](
            x, w, b, out,
            N, IC,
            IH, IW,
            OC,
            OH, OW,
            POH, POW,
            KH, KW,
            POOL,
            self.subtract1_value, self.subtract2_value,
        )
        return out