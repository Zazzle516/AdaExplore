import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_tanh_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, OH, OW,
    POH, POW,
    sub1, sub2,
    stride_xn, stride_xh, stride_xw, stride_xc,  # NHWC strides
    stride_wo, stride_wk,  # weight [OC, IC*KH*KW]
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IC_C: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    PK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    # Each program handles BLOCK_SP pooled output positions for BLOCK_OC channels
    # Total conv output positions = BLOCK_SP * PK * PK
    SP_CONV: tl.constexpr = BLOCK_SP * PK * PK
    K_TOTAL: tl.constexpr = IC_C * KH * KW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    oc_mask = oc_offs < OC

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP] pooled idx
    sp_mask = sp_offs < (POH * POW)

    # pooled coord
    poh = sp_offs // POW
    pow_ = sp_offs % POW

    # For each pooled pos, we have PK*PK conv positions
    # Build a [BLOCK_SP, PK*PK] grid of (oh, ow) for conv output positions
    pp = tl.arange(0, PK * PK)  # [PK*PK]
    pi = pp // PK
    pj = pp % PK

    # conv output coords: [BLOCK_SP, PK*PK]
    oh_grid = poh[:, None] * PK + pi[None, :]  # [BLOCK_SP, PK*PK]
    ow_grid = pow_[:, None] * PK + pj[None, :]  # [BLOCK_SP, PK*PK]

    # Flatten conv positions: total = BLOCK_SP * PK * PK
    oh_flat = tl.reshape(oh_grid, (SP_CONV,))
    ow_flat = tl.reshape(ow_grid, (SP_CONV,))
    sp_flat_mask = tl.reshape(tl.broadcast_to(sp_mask[:, None], (BLOCK_SP, PK * PK)), (SP_CONV,))

    # Accumulator: [BLOCK_OC, SP_CONV]
    acc = tl.zeros((BLOCK_OC, SP_CONV), dtype=tl.float32)

    # K loop: ic * KH * KW
    k_offs = tl.arange(0, BLOCK_K)  # [BLOCK_K]

    for k_start in range(0, K_TOTAL, BLOCK_K):
        k_cur = k_start + k_offs  # [BLOCK_K]
        k_mask = k_cur < K_TOTAL

        # decompose k -> (ic, kh, kw)
        ic = k_cur // (KH * KW)
        khkw = k_cur % (KH * KW)
        kh = khkw // KW
        kw = khkw % KW

        # weight: [OC, K] => [BLOCK_OC, BLOCK_K]
        w_off = oc_offs[:, None] * stride_wo + k_cur[None, :] * stride_wk
        w_m = oc_mask[:, None] & k_mask[None, :]
        w_val = tl.load(w_ptr + w_off, mask=w_m, other=0.0)  # [BLOCK_OC, BLOCK_K]

        # input: NHWC, x[n, oh+kh, ow+kw, ic]
        # need [BLOCK_K, SP_CONV]
        ih = oh_flat[None, :] + kh[:, None]  # [BLOCK_K, SP_CONV]
        iw = ow_flat[None, :] + kw[:, None]  # [BLOCK_K, SP_CONV]
        x_off = (pid_n * stride_xn
                 + ih * stride_xh
                 + iw * stride_xw
                 + ic[:, None] * stride_xc)
        x_m = k_mask[:, None] & sp_flat_mask[None, :] & (ih < H) & (iw < W)
        x_val = tl.load(x_ptr + x_off, mask=x_m, other=0.0)  # [BLOCK_K, SP_CONV]

        acc += tl.dot(w_val, x_val)

    # bias
    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc += b_val[:, None]

    # sub1, tanh, sub2
    v = acc - sub1
    # tanh
    e2 = tl.exp(2.0 * v)
    t = (e2 - 1.0) / (e2 + 1.0)
    v = t - sub2

    # Reshape to [BLOCK_OC, BLOCK_SP, PK*PK] and reduce
    v_r = tl.reshape(v, (BLOCK_OC, BLOCK_SP, PK * PK))
    pooled = tl.sum(v_r, axis=2) / (PK * PK)  # [BLOCK_OC, BLOCK_SP]

    # store: out is [N, OC, POH, POW] contiguous
    out_off = (pid_n * OC * POH * POW
               + oc_offs[:, None] * POH * POW
               + sp_offs[None, :])
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, pooled, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = float(subtract1_value)
        self.subtract2_value = float(subtract2_value)
        self.kernel_size = kernel_size
        self.kernel_size_pool = kernel_size_pool
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Pre-permute weight to [OC, IC*KH*KW] (flatten K dimension w/ IC outermost)
        # Conv weight layout: [OC, IC, KH, KW] contiguous => flatten matches K = ic*KH*KW + kh*KW + kw
        w = self.conv.weight.detach().contiguous()
        self.register_buffer('w_flat', w.view(out_channels, -1).contiguous())
        self.register_buffer('b_flat', self.conv.bias.detach().contiguous())

    def forward(self, x):
        x = x.contiguous().cuda()
        # Permute to NHWC
        N, IC, H, W = x.shape
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1
        PK = self.kernel_size_pool
        POH = OH // PK
        POW = OW // PK

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=x.dtype)

        w = self.w_flat.cuda()
        b = self.b_flat.cuda()

        # NHWC strides: [H*W*IC, W*IC, IC, 1]
        stride_xn = H * W * IC
        stride_xh = W * IC
        stride_xw = IC
        stride_xc = 1

        stride_wo = IC * KH * KW
        stride_wk = 1

        BLOCK_OC = 64
        BLOCK_SP = 64  # pooled positions per program
        BLOCK_K = 32

        grid = (
            N,
            (OC + BLOCK_OC - 1) // BLOCK_OC,
            (POH * POW + BLOCK_SP - 1) // BLOCK_SP,
        )

        fused_conv_tanh_pool_kernel[grid](
            x_nhwc, w, b, out,
            N, IC, H, W,
            OC, OH, OW,
            POH, POW,
            self.subtract1_value, self.subtract2_value,
            stride_xn, stride_xh, stride_xw, stride_xc,
            stride_wo, stride_wk,
            BLOCK_OC=BLOCK_OC,
            BLOCK_SP=BLOCK_SP,
            BLOCK_K=BLOCK_K,
            IC_C=IC,
            KH=KH,
            KW=KW,
            PK=PK,
            num_warps=4,
            num_stages=3,
        )
        return out